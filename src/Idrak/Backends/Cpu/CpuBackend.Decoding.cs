// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Runtime.InteropServices;

namespace Idrak.Backends.Cpu;

// Fused inference kernels and incremental-decoding primitives (KV cache, masks, on-device sampling).
internal sealed partial class CpuBackend
{
    public override void DownloadRange(Storage source, int offset, Span<float> destination) =>
        D(source).AsSpan(offset, destination.Length).CopyTo(destination);

    public override void ScaleMaskSoftmax(Storage x, Storage? mask, Storage y, int rows, int cols, int maskRows, float scale)
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
                float max = float.NegativeInfinity;
                for (int j = 0; j < cols; j++)
                {
                    ys[j] = xs[j] * scale + (mv is null ? 0f : ms[j]);
                    max = MathF.Max(max, ys[j]);
                }

                CpuMath.Scale(ys, 1f / CpuMath.ExpShifted(ys, max));
            }
        });
    }

    public override void LayerNormTrain(Storage x, Storage gamma, Storage beta, Storage y, Storage stats, int rows, int cols, float eps)
    {
        LayerNormFused(x, gamma, beta, y, rows, cols, eps);
        float[] xv = D(x), sv = D(stats);
        For(rows, (long)rows * cols * 2, (start, end) =>
        {
            for (int r = start; r < end; r++)
            {
                var xs = xv.AsSpan(r * cols, cols);
                float mean = 0f;
                foreach (float v in xs)
                {
                    mean += v;
                }

                mean /= cols;
                float var = 0f;
                foreach (float v in xs)
                {
                    var += (v - mean) * (v - mean);
                }

                sv[r] = mean;
                sv[rows + r] = 1f / MathF.Sqrt(var / cols + eps);
            }
        });
    }

    public override void LayerNormBackward(Storage x, Storage gamma, Storage dy, Storage stats, Storage? dx, Storage? dgamma, Storage? dbeta, int rows, int cols)
    {
        float[] xv = D(x), gv = D(gamma), dyv = D(dy), sv = D(stats);
        float[]? dxv = dx is null ? null : D(dx);
        if (dxv is not null)
        {
            For(rows, (long)rows * cols * 4, (start, end) =>
            {
                for (int r = start; r < end; r++)
                {
                    float mean = sv[r], rstd = sv[rows + r], s1 = 0f, s2 = 0f;
                    for (int j = 0; j < cols; j++)
                    {
                        float xhat = (xv[r * cols + j] - mean) * rstd, g = dyv[r * cols + j] * gv[j];
                        s1 += g;
                        s2 += g * xhat;
                    }

                    s1 /= cols;
                    s2 /= cols;
                    for (int j = 0; j < cols; j++)
                    {
                        float xhat = (xv[r * cols + j] - mean) * rstd, g = dyv[r * cols + j] * gv[j];
                        dxv[r * cols + j] += rstd * (g - s1 - xhat * s2);
                    }
                }
            });
        }

        if (dgamma is not null || dbeta is not null)
        {
            float[]? dg = dgamma is null ? null : D(dgamma), db = dbeta is null ? null : D(dbeta);
            For(cols, (long)rows * cols * 2, (start, end) =>
            {
                for (int j = start; j < end; j++)
                {
                    double sg = 0, sb = 0;
                    for (int r = 0; r < rows; r++)
                    {
                        float d = dyv[r * cols + j];
                        sg += d * ((xv[r * cols + j] - sv[r]) * sv[rows + r]);
                        sb += d;
                    }

                    if (dg is not null)
                    {
                        dg[j] += (float)sg;
                    }

                    if (db is not null)
                    {
                        db[j] += (float)sb;
                    }
                }
            });
        }
    }

    public override void LayerNormFused(Storage x, Storage gamma, Storage beta, Storage y, int rows, int cols, float eps)
    {
        float[] xv = D(x), gv = D(gamma), bv = D(beta), yv = D(y);
        For(rows, (long)rows * cols * 4, (start, end) =>
        {
            for (int r = start; r < end; r++)
            {
                var xs = xv.AsSpan(r * cols, cols);
                var ys = yv.AsSpan(r * cols, cols);
                float mean = 0f;
                foreach (float v in xs)
                {
                    mean += v;
                }

                mean /= cols;
                float var = 0f;
                foreach (float v in xs)
                {
                    var += (v - mean) * (v - mean);
                }

                float inv = 1f / MathF.Sqrt(var / cols + eps);
                for (int j = 0; j < cols; j++)
                {
                    ys[j] = (xs[j] - mean) * inv * gv[j] + bv[j];
                }
            }
        });
    }

    public override void BiasGelu(Storage x, Storage bias, Storage y, int n, int cols)
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
                    yv[o + j] = 0.5f * v * (1f + MathF.Tanh(GeluK * (v + GeluC * v * v * v)));
                }
            }
        });
    }

    public override void DecoderMask(Storage position, Storage mask, int rows, int capacity)
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

    public override void KeyValueWrite(Storage source, Storage cache, Storage position, int heads, int steps, int capacity, int dim)
    {
        int pos = (int)D(position)[0];
        float[] sv = D(source), cv = D(cache);
        for (int h = 0; h < heads; h++)
        {
            sv.AsSpan(h * steps * dim, steps * dim).CopyTo(cv.AsSpan((h * capacity + pos) * dim, steps * dim));
        }
    }

    public override void SoftmaxCrossEntropyRows(Storage logits, Storage targets, Storage weights, Storage losses, int rows, int vocabulary, float scale)
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
                    sum += Math.Exp(v - max);
                }

                float lse = max + (float)Math.Log(sum);
                lv[r] = w * (lse - x[target]);
                float g = scale * w;
                for (int j = 0; j < x.Length; j++)
                {
                    x[j] = g * (MathF.Exp(x[j] - lse) - (j == target ? 1f : 0f));
                }
            }
        });
    }

    public override void KeyValueWriteBFloat16(Storage source, Storage cache, Storage position, int heads, int steps, int capacity, int dim)
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
                    row[d] = Layers.BFloat16Weight.Round(sv[(h * steps + t) * dim + d]);
                }
            }
        }
    }

    public override void AttentionBFloat16(Storage q, Storage keys, Storage values, Storage position, Storage y, int heads, int rowsPerHead,
        int steps, int capacity, int dim, float scale, bool tiled)
    {
        // Straight from the bfloat16 cache: each cached key and value row is widened into a small buffer as it is used,
        // in parallel over the query rows (no float copy of the whole filled cache per call).
        float[] qv = D(q), yv = D(y);
        int position0 = (int)D(position)[0], stride = (dim + 1) / 2 * 2;
        For(heads * rowsPerHead, (long)heads * rowsPerHead * dim * Math.Max(1, position0), (first, last) =>
        {
            var kh = MemoryMarshal.Cast<float, ushort>(D(keys).AsSpan());
            var vh = MemoryMarshal.Cast<float, ushort>(D(values).AsSpan());
            var scores = ArrayPool<float>.Shared.Rent(capacity);
            var widened = ArrayPool<float>.Shared.Rent(dim);
            var row = widened.AsSpan(0, dim);
            for (int r = first; r < last; r++)
            {
                int h = r / rowsPerHead, count = Math.Min(position0 + r % rowsPerHead % steps, capacity - 1) + 1;
                var query = qv.AsSpan(r * dim, dim);
                float max = float.NegativeInfinity;
                for (int c = 0; c < count; c++)
                {
                    WidenBFloat16(kh.Slice((h * capacity + c) * stride, dim), row);
                    scores[c] = Dot(query, row) * scale;
                    max = MathF.Max(max, scores[c]);
                }

                float sum = CpuMath.ExpShifted(scores.AsSpan(0, count), max);
                var output = yv.AsSpan(r * dim, dim);
                output.Clear();
                for (int c = 0; c < count; c++)
                {
                    WidenBFloat16(vh.Slice((h * capacity + c) * stride, dim), row);
                    AddScaled(output, row, scores[c] / sum);
                }
            }

            ArrayPool<float>.Shared.Return(scores);
            ArrayPool<float>.Shared.Return(widened);
        });
    }

    public override void SampleRows(Storage logits, Storage ids, Storage stats, Storage step, int rows, int vocabulary,
        int rowStride, int rowOffset, float temperature, int topK, float topP, float minP, uint seed)
    {
        float[] lv = D(logits), iv = D(ids), sv = D(stats);
        uint stepNumber = (uint)D(step)[0];
        float invT = 1f / MathF.Max(temperature, 1e-3f);
        float[] e = ArrayPool<float>.Shared.Rent(vocabulary);
        float[] top = ArrayPool<float>.Shared.Rent(Math.Max(1, Math.Min(topK, vocabulary)));
        float[]? candidateScores = null, candidateWeights = null;
        Span<int> taken = stackalloc int[5];
        for (int r = 0; r < rows; r++)
        {
            var z = lv.AsSpan(r * rowStride + rowOffset, vocabulary);
            float max = float.NegativeInfinity;
            foreach (float v in z)
            {
                max = MathF.Max(max, v * invT);
            }

            // Top-k: the k-th largest distinct scaled score is the cut-off (ties at the cut-off are kept); fewer than k
            // distinct scores keep everything. One pass keeping the k largest distinct scores, largest first.
            float threshold = float.NegativeInfinity;
            if (topK > 0 && topK < vocabulary)
            {
                int count = 0;
                foreach (float v in z)
                {
                    float sc = v * invT;
                    if (float.IsNaN(sc) || count == topK && !(sc > top[count - 1]))
                    {
                        continue;
                    }

                    int at = count;
                    while (at > 0 && top[at - 1] < sc)
                    {
                        at--;
                    }

                    if (at > 0 && top[at - 1] == sc)
                    {
                        continue;                                            // already kept
                    }

                    int last = Math.Min(count, topK - 1);
                    for (int i = last; i > at; i--)
                    {
                        top[i] = top[i - 1];
                    }

                    top[at] = sc;
                    count = Math.Min(count + 1, topK);
                }

                threshold = count == topK ? top[topK - 1] : float.NegativeInfinity;
            }

            // Min-p: keep tokens at least minP times as likely as the best: s >= max + ln(minP).
            if (minP > 0f)
            {
                threshold = MathF.Max(threshold, max + MathF.Log(minP));
            }

            // Top-p (nucleus): the highest cut-off whose kept mass is still >= topP of the total, by bisection. Only scores at
            // or above the bisection's lower bound can count, so their weights are computed once and summed in token order.
            if (topP > 0f && topP < 1f)
            {
                float total = 0f, floor = MathF.Max(max - 40f, threshold);
                candidateScores ??= ArrayPool<float>.Shared.Rent(vocabulary);
                candidateWeights ??= ArrayPool<float>.Shared.Rent(vocabulary);
                int candidates = 0;
                foreach (float v in z)
                {
                    float sc = v * invT;
                    if (sc >= threshold)
                    {
                        float w = MathF.Exp(sc - max);
                        total += w;
                        if (sc >= floor)
                        {
                            candidateScores[candidates] = sc;
                            candidateWeights[candidates++] = w;
                        }
                    }
                }

                float goal = total * topP, lo = floor, hi = max;
                for (int it = 0; it < 24; it++)
                {
                    float mid = (lo + hi) * 0.5f, mass = 0f;
                    for (int c = 0; c < candidates; c++)
                    {
                        mass += candidateScores[c] >= mid ? candidateWeights[c] : 0f;
                    }

                    if (mass >= goal)
                    {
                        // Every later midpoint is at least this one, so scores below it never count again: drop them (in
                        // order, so the sums add the same values in the same order; the dropped ones only added zeros).
                        lo = mid;
                        int kept = 0;
                        for (int c = 0; c < candidates; c++)
                        {
                            if (candidateScores[c] >= mid)
                            {
                                candidateScores[kept] = candidateScores[c];
                                candidateWeights[kept++] = candidateWeights[c];
                            }
                        }

                        candidates = kept;
                    }
                    else
                    {
                        hi = mid;
                    }
                }

                threshold = lo;
            }

            float sum = 0f;
            for (int j = 0; j < vocabulary; j++)
            {
                float sc = z[j] * invT;
                e[j] = sc >= threshold ? MathF.Exp(sc - max) : 0f;
                sum += e[j];
            }

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
                    entropy -= p * MathF.Log2(p);
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

            iv[r] = chosen;
            int o = ((int)stepNumber * rows + r) * 13;
            sv[o] = chosen;
            sv[o + 1] = e[chosen] / sum;
            sv[o + 2] = entropy;
            for (int a = 0; a < 5; a++)
            {
                int best = a < found ? taken[a] : -1;
                sv[o + 3 + 2 * a] = best;
                sv[o + 4 + 2 * a] = best < 0 ? 0f : e[best] / sum;
            }
        }

        ArrayPool<float>.Shared.Return(e);
        ArrayPool<float>.Shared.Return(top);
        if (candidateScores is not null)
        {
            ArrayPool<float>.Shared.Return(candidateScores);
            ArrayPool<float>.Shared.Return(candidateWeights!);
        }
    }

    public override void PenalizeRows(Storage logits, Storage work, Storage history, Storage length, int rows, int vocabulary,
        int rowStride, int rowOffset, int capacity, int lastN, float repeat, float presence, float frequency)
    {
        float[] lv = D(logits), wv = D(work), hv = D(history);
        uint len = (uint)D(length)[0];
        int n = (int)Math.Min(len, (uint)Math.Min(lastN, capacity));
        for (int r = 0; r < rows; r++)
        {
            var w = wv.AsSpan(r * vocabulary, vocabulary);
            lv.AsSpan(r * rowStride + rowOffset, vocabulary).CopyTo(w);
            for (int k = 0; k < n; k++)
            {
                int id = (int)hv[r * capacity + (int)((len - 1 - (uint)k) % (uint)capacity)];
                bool seenMoreRecently = false;
                int count = 0;
                for (int q = 0; q < n; q++)
                {
                    int other = (int)hv[r * capacity + (int)((len - 1 - (uint)q) % (uint)capacity)];
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

    public override void HistoryPush(Storage ids, Storage history, Storage length, int rows, int capacity)
    {
        float[] iv = D(ids), hv = D(history);
        uint len = (uint)D(length)[0];
        for (int r = 0; r < rows; r++)
        {
            hv[r * capacity + (int)(len % (uint)capacity)] = iv[r];
        }
    }
}
