// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Idrak.Abstraction.Devices.Cpu;

// Fused inference kernels and incremental-decoding primitives (KV cache, masks, on-device sampling).
internal sealed partial class CpuBackend
{
    public override void DownloadRange(Storage source, int offset, Span<float> destination)
    {
        ThrowIfDisposed();
        if (source is MappedStorage mapped)
        {
            CopyOutOfMapping(mapped, offset, destination, "downloaded (read whole, moved to another device)");
            return;
        }

        D(source).AsSpan(offset, destination.Length).CopyTo(destination);
    }

    public override void ScaleMaskSoftmaxKernel(Storage x, Storage? mask, Storage y, int rows, int cols, int maskRows, float scale)
    {
        float[] xv = D(x), yv = D(y);
        float[]? mv = mask is null ? null : D(mask);
        For(rows, (long)rows * cols * 8, (start, end) =>
        {
            for (int r = start; r < end; r++)
            {
                var xs = xv.AsSpan(r * cols, cols);
                var ys = yv.AsSpan(r * cols, cols);
                var ms = mv is null ? default : mv.AsSpan(r % maskRows * cols, cols);
                float max = CpuMath.ScaleAddMax(xs, ms, ys, scale);
                CpuMath.Scale(ys, 1f / CpuMath.ExpShifted(ys, max));
            }
        });
    }

    public override void LayerNormTrainKernel(Storage x, Storage gamma, Storage beta, Storage y, Storage stats, int rows, int cols, float eps)
    {
        LayerNormFused(x, gamma, beta, y, rows, cols, eps);
        float[] xv = D(x), sv = D(stats);
        For(rows, (long)rows * cols * 2, (start, end) =>
        {
            for (int r = start; r < end; r++)
            {
                var xs = xv.AsSpan(r * cols, cols);
                float mean = RowSum(xs) / cols;
                float var = RowSquaredDeviation(xs, mean);
                sv[r] = mean;
                sv[rows + r] = 1f / MathF.Sqrt(var / cols + eps);
            }
        });
    }

    public override void LayerNormBackwardKernel(Storage x, Storage gamma, Storage dy, Storage stats, Storage? dx, Storage? dgamma, Storage? dbeta, int rows, int cols)
    {
        float[] xv = D(x), gv = D(gamma), dyv = D(dy), sv = D(stats);
        float[]? dxv = dx is null ? null : D(dx);
        if (dxv is not null)
        {
            For(rows, (long)rows * cols * 4, (start, end) =>
            {
                for (int r = start; r < end; r++)
                {
                    float mean = sv[r], rstd = sv[rows + r];
                    LayerNormInputGradient(xv.AsSpan(r * cols, cols), gv, dyv.AsSpan(r * cols, cols), dxv.AsSpan(r * cols, cols), mean, rstd);
                }
            });
        }

        if (dgamma is not null || dbeta is not null)
        {
            float[]? dg = dgamma is null ? null : D(dgamma), db = dbeta is null ? null : D(dbeta);
            // Each worker's columns walked row by row (memory order), each column's sums still over the rows in order.
            For(cols, (long)rows * cols * 2, (start, end) =>
            {
                int width = end - start;
                var sums = ArrayPool<double>.Shared.Rent(2 * width);
                try
                {
                    var sg = sums.AsSpan(0, width);
                    var sb = sums.AsSpan(width, width);
                    sg.Clear();
                    sb.Clear();
                    for (int r = 0; r < rows; r++)
                    {
                        var ds = dyv.AsSpan(r * cols + start, width);
                        var xs = xv.AsSpan(r * cols + start, width);
                        float mean = sv[r], rstd = sv[rows + r];
                        for (int j = 0; j < width; j++)
                        {
                            float d = ds[j];
                            sg[j] += d * ((xs[j] - mean) * rstd);
                            sb[j] += d;
                        }
                    }

                    for (int j = 0; j < width; j++)
                    {
                        if (dg is not null)
                        {
                            dg[start + j] += (float)sg[j];
                        }

                        if (db is not null)
                        {
                            db[start + j] += (float)sb[j];
                        }
                    }
                }
                finally
                {
                    ArrayPool<double>.Shared.Return(sums);
                }
            });
        }
    }

    public override void LayerNormFusedKernel(Storage x, Storage gamma, Storage beta, Storage y, int rows, int cols, float eps)
    {
        float[] xv = D(x), gv = D(gamma), bv = D(beta), yv = D(y);
        For(rows, (long)rows * cols * 4, (start, end) =>
        {
            for (int r = start; r < end; r++)
            {
                var xs = xv.AsSpan(r * cols, cols);
                var ys = yv.AsSpan(r * cols, cols);
                float mean = RowSum(xs) / cols;
                float var = RowSquaredDeviation(xs, mean);
                float inv = 1f / MathF.Sqrt(var / cols + eps);
                NormalizeRow(xs, gv, bv, ys, mean, inv);
            }
        });
    }

    // y = (x − mean) · inv · gamma + beta over a row (SIMD, scalar tail).
    private static void NormalizeRow(ReadOnlySpan<float> x, ReadOnlySpan<float> gamma, ReadOnlySpan<float> beta, Span<float> y, float mean, float inv)
    {
        var xv = MemoryMarshal.Cast<float, Vector<float>>(x);
        var gv = MemoryMarshal.Cast<float, Vector<float>>(gamma);
        var bv = MemoryMarshal.Cast<float, Vector<float>>(beta);
        var yv = MemoryMarshal.Cast<float, Vector<float>>(y);
        var m = new Vector<float>(mean);
        var s = new Vector<float>(inv);
        for (int i = 0; i < xv.Length; i++)
        {
            yv[i] = Vector.FusedMultiplyAdd((xv[i] - m) * s, gv[i], bv[i]);
        }

        for (int j = xv.Length * Vector<float>.Count; j < x.Length; j++)
        {
            y[j] = MathF.FusedMultiplyAdd((x[j] - mean) * inv, gamma[j], beta[j]);
        }
    }

    // dx += rstd · (g − mean(g) − xhat · mean(g · xhat)) over a row, g = dy · gamma, xhat = (x − mean) · rstd (SIMD, scalar tail).
    private static void LayerNormInputGradient(ReadOnlySpan<float> x, ReadOnlySpan<float> gamma, ReadOnlySpan<float> dy, Span<float> dx, float mean, float rstd)
    {
        int cols = x.Length;
        var xv = MemoryMarshal.Cast<float, Vector<float>>(x);
        var gv = MemoryMarshal.Cast<float, Vector<float>>(gamma);
        var dyv = MemoryMarshal.Cast<float, Vector<float>>(dy);
        var dxv = MemoryMarshal.Cast<float, Vector<float>>(dx);
        var m = new Vector<float>(mean);
        var r = new Vector<float>(rstd);
        int tail = xv.Length * Vector<float>.Count;
        var (s1, s2) = CpuMath.FoldPair(new LayerNormGradientTerms(ref MemoryMarshal.GetReference(x), ref MemoryMarshal.GetReference(gamma),
            ref MemoryMarshal.GetReference(dy), mean, rstd), cols);                // Σ g and Σ g · xhat, in the fixed lanes
        s1 /= cols;
        s2 /= cols;
        var v1 = new Vector<float>(s1);
        var v2 = new Vector<float>(s2);
        for (int i = 0; i < xv.Length; i++)
        {
            var g = dyv[i] * gv[i];
            var xhat = (xv[i] - m) * r;
            dxv[i] = Vector.FusedMultiplyAdd(r, g - v1 - xhat * v2, dxv[i]);
        }

        for (int j = tail; j < cols; j++)
        {
            float g = dy[j] * gamma[j], xhat = (x[j] - mean) * rstd;
            dx[j] = MathF.FusedMultiplyAdd(rstd, g - s1 - xhat * s2, dx[j]);
        }
    }

    // g = dy · gamma into Σ g, and g · xhat (xhat = (x − mean) · rstd, one FMA) into the second sum.
    private readonly ref struct LayerNormGradientTerms : CpuMath.ILanePair
    {
        private readonly ref float _x, _gamma, _dy;
        private readonly float _mean, _rstd;

        public LayerNormGradientTerms(ref float x, ref float gamma, ref float dy, float mean, float rstd)
        {
            _x = ref x;
            _gamma = ref gamma;
            _dy = ref dy;
            _mean = mean;
            _rstd = rstd;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Add(ref Vector<float> first, ref Vector<float> second, int i)
        {
            var g = Vector.LoadUnsafe(ref _dy, (nuint)i) * Vector.LoadUnsafe(ref _gamma, (nuint)i);
            first += g;
            second = Vector.FusedMultiplyAdd(g, (Vector.LoadUnsafe(ref _x, (nuint)i) - new Vector<float>(_mean)) * new Vector<float>(_rstd), second);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Add(ref float first, ref float second, int i)
        {
            float g = Unsafe.Add(ref _dy, i) * Unsafe.Add(ref _gamma, i);
            first += g;
            second = MathF.FusedMultiplyAdd(g, (Unsafe.Add(ref _x, i) - _mean) * _rstd, second);
        }
    }

    public override void BiasGeluKernel(Storage x, Storage bias, Storage y, int n, int cols)
    {
        float[] xv = D(x), bv = D(bias), yv = D(y);
        For(n / cols, n * 8L, (start, end) =>
        {
            for (int r = start; r < end; r++)
            {
                int o = r * cols;
                for (int j = 0; j < cols; j++)
                {
                    float v = xv[o + j] + bv[j];
                    yv[o + j] = 0.5f * v * (1f + CpuMath.Tanh(GeluK * (v + GeluC * v * v * v)));
                }
            }
        });
    }

    public override void DecoderMaskKernel(Storage position, Storage mask, int rows, int capacity)
    {
        int pos = (int)D(position)[0];
        float[] mv = D(mask);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < capacity; j++)
            {
                mv[i * capacity + j] = j <= pos + i ? 0f : -1e9f;
            }
        }
    }

    public override void KeyValueWriteKernel(Storage source, Storage cache, Storage position, int heads, int steps, int capacity, int dim)
    {
        int pos = (int)D(position)[0];
        float[] sv = D(source), cv = D(cache);
        for (int h = 0; h < heads; h++)
        {
            sv.AsSpan(h * steps * dim, steps * dim).CopyTo(cv.AsSpan((h * capacity + pos) * dim, steps * dim));
        }
    }

    public override void SoftmaxCrossEntropyRowsKernel(Storage logits, Storage targets, Storage weights, Storage losses, int rows, int vocabulary, float scale)
    {
        float[] xv = D(logits), tv = D(targets), wv = D(weights), lv = D(losses);
        For(rows, (long)rows * vocabulary * 4, (first, last) =>
        {
            for (int r = first; r < last; r++)
            {
                var x = xv.AsSpan(r * vocabulary, vocabulary);
                int target = (int)tv[r];
                float max = CpuMath.Max(x), w = wv[r];
                double sum = 0;
                foreach (float v in x)
                {
                    sum += CpuMath.ExpDouble(v - max);
                }

                float lse = max + (float)CpuMath.LogDouble(sum);
                lv[r] = w * (lse - x[target]);
                float g = scale * w;
                for (int j = 0; j < x.Length; j++)
                {
                    x[j] = g * (CpuMath.Exp(x[j] - lse) - (j == target ? 1f : 0f));
                }
            }
        });
    }

    public override void KeyValueWriteBFloat16Kernel(Storage source, Storage cache, Storage position, int heads, int steps, int capacity, int dim)
    {
        int pos = (int)D(position)[0], stride = (dim + 1) / 2 * 2;
        float[] sv = D(source);
        var halves = MemoryMarshal.Cast<float, ushort>(D(cache).AsSpan());
        for (int h = 0; h < heads; h++)
        {
            for (int t = 0; t < steps; t++)
            {
                var row = halves.Slice(((h * capacity) + pos + t) * stride, stride);
                row.Clear();
                for (int d = 0; d < dim; d++)
                {
                    row[d] = BFloat16Bits.Round(sv[(h * steps + t) * dim + d]);
                }
            }
        }
    }

    public override void SampleRowsKernel(Storage logits, Storage ids, Storage stats, Storage step, int rows, int vocabulary,
        int rowStride, int rowOffset, float temperature, int topK, float topP, float minP, uint seed)
    {
        float[] lv = D(logits), sv = D(stats);
        var iv = I(ids);                                                    // int32 token ids (plan 37, T1)
        uint stepNumber = (uint)D(step)[0];
        float[] e = ArrayPool<float>.Shared.Rent(vocabulary);
        float[] top = ArrayPool<float>.Shared.Rent(Math.Max(1, Math.Min(topK, vocabulary)));
        float[]? candidateScores = null, candidateWeights = null;
        Span<int> taken = stackalloc int[5];
        try
        {
            for (int r = 0; r < rows; r++)
            {
                var z = lv.AsSpan(r * rowStride + rowOffset, vocabulary);
                float sum = Idrak.Abstraction.Generation.SamplingWeights.Compute(z, e.AsSpan(0, vocabulary), temperature, topK, topP, minP, top, ref candidateScores, ref candidateWeights);
                float target = CounterRandom.Uniform(seed, stepNumber, (uint)r) * sum;
                int chosen = -1;
                float cumulative = 0f;
                for (int j = 0; j < vocabulary; j++)
                {
                    if (e[j] > 0f)
                    {
                        chosen = j;
                        cumulative += e[j];
                        if (cumulative > target)
                        {
                            break;
                        }
                    }
                }

                // Entropy, and the five most likely tokens (largest weight first, the lower id on ties) in the same pass.
                float entropy = 0f;
                int found = 0;
                for (int j = 0; j < vocabulary; j++)
                {
                    float w = e[j];
                    if (w > 0f)
                    {
                        float p = w / sum;
                        entropy -= p * (CpuMath.Log(p) * 1.44269504f);   // log2 p as ln p · log2 e, no C runtime's logarithm
                        if (found < 5 || w > e[taken[4]])
                        {
                            int at = Math.Min(found, 4);
                            while (at > 0 && e[taken[at - 1]] < w)
                            {
                                at--;
                            }

                            for (int i = Math.Min(found, 4); i > at; i--)
                            {
                                taken[i] = taken[i - 1];
                            }

                            taken[at] = j;
                            found = Math.Min(found + 1, 5);
                        }
                    }
                }

                iv[r] = Math.Max(chosen, 0);                                     // nothing to sample: a valid id for the next step
                int o = ((int)stepNumber * rows + r) * 13;
                sv[o] = chosen;
                sv[o + 1] = chosen < 0 ? 0f : e[chosen] / sum;
                sv[o + 2] = entropy;
                for (int a = 0; a < 5; a++)
                {
                    int best = a < found ? taken[a] : -1;
                    sv[o + 3 + 2 * a] = best;
                    sv[o + 4 + 2 * a] = best < 0 ? 0f : e[best] / sum;
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(e);
            ArrayPool<float>.Shared.Return(top);
            if (candidateScores is not null)
            {
                ArrayPool<float>.Shared.Return(candidateScores);
                ArrayPool<float>.Shared.Return(candidateWeights!);
            }
        }
    }

    public override void PenalizeRowsKernel(Storage logits, Storage work, Storage history, Storage length, int rows, int vocabulary,
        int rowStride, int rowOffset, int capacity, int lastN, float repeat, float presence, float frequency)
    {
        float[] lv = D(logits), wv = D(work);
        var hv = I(history);
        uint len = (uint)D(length)[0];
        int n = (int)Math.Min(len, (uint)Math.Min(lastN, capacity));
        for (int r = 0; r < rows; r++)
        {
            var w = wv.AsSpan(r * vocabulary, vocabulary);
            lv.AsSpan(r * rowStride + rowOffset, vocabulary).CopyTo(w);
            for (int k = 0; k < n; k++)
            {
                int id = hv[r * capacity + (int)((len - 1 - (uint)k) % (uint)capacity)];
                bool seenMoreRecently = false;
                int count = 0;
                for (int q = 0; q < n; q++)
                {
                    int other = hv[r * capacity + (int)((len - 1 - (uint)q) % (uint)capacity)];
                    if (other == id)
                    {
                        count++;
                        seenMoreRecently |= q < k;
                    }
                }

                if (seenMoreRecently || (uint)id >= (uint)vocabulary)
                {
                    continue;
                }

                float x = w[id];
                x = x > 0f ? x / repeat : x * repeat;
                w[id] = x - presence - frequency * count;
            }
        }
    }

    public override void HistoryPushKernel(Storage ids, Storage history, Storage length, int rows, int capacity)
    {
        Span<int> iv = I(ids), hv = I(history);
        uint len = (uint)D(length)[0];
        for (int r = 0; r < rows; r++)
        {
            hv[r * capacity + (int)(len % (uint)capacity)] = iv[r];
        }
    }
}
