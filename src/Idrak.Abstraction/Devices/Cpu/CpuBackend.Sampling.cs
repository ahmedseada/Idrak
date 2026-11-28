// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Runtime.CompilerServices;
using Idrak.Abstraction.Generation;

namespace Idrak.Abstraction.Devices.Cpu;

// Sampling and the row losses: a row's draw from the sampler's weights (SamplingWeights, the rule the host's checks read),
// the repetition penalties (TokenSampler.Penalize's rule), the history ring, the best k log-probabilities and the token
// cross-entropy. Indices and the device-side counters are read and written as their storages' types hold them.
internal sealed unsafe partial class CpuBackend
{
    public override void SampleRowsKernel(Storage logits, Storage ids, Storage stats, Storage step, int rows, int vocabulary,
        int rowStride, int rowOffset, float temperature, int topK, float topP, float minP, uint seed)
    {
        var wz = D(logits);
        var tz = new TypedRanges(logits, 0, "SampleRows");                    // a typed row read through its type when drawn
        var sv = V(stats);
        fixed (float* pi = D(ids), pstep = D(step))
        {
            var into = IndicesOf(ids, pi);
            uint stepNumber = (uint)IndicesOf(step, pstep)[0];
            float[] e = ArrayPool<float>.Shared.Rent(Math.Max(1, vocabulary));
            float[] top = ArrayPool<float>.Shared.Rent(Math.Max(1, Math.Min(Math.Max(topK, 1), vocabulary)));
            float[]? candidateScores = null, candidateWeights = null;
            Span<int> taken = stackalloc int[5];
            try
            {
                for (int r = 0; r < rows; r++)
                {
                    long first = (long)r * rowStride + rowOffset;
                    var z = tz.Values(wz, first, vocabulary, out long shift);
                    var row = z.AsSpan((int)(shift + first), vocabulary);
                    var weights = e.AsSpan(0, vocabulary);
                    float sum = SamplingWeights.Compute(row, weights, temperature, topK, topP, minP, top, ref candidateScores, ref candidateWeights);
                    long o = ((long)stepNumber * rows + r) * 13;
                    if (!(sum > 0f))
                    {
                        into.Set(r, 0);
                        sv[o] = -1f;
                        sv[o + 1] = 0f;
                        sv[o + 2] = 0f;
                        for (int a = 0; a < 5; a++)
                        {
                            sv[o + 3 + 2 * a] = -1f;
                            sv[o + 4 + 2 * a] = 0f;
                        }

                        continue;
                    }

                    float target = CounterRandom.Uniform(seed, stepNumber, (uint)r) * sum;
                    int chosen = -1;
                    float cumulative = 0f;
                    for (int j = 0; j < vocabulary; j++)
                    {
                        if (weights[j] > 0f)
                        {
                            chosen = j;
                            cumulative += weights[j];
                            if (cumulative > target)
                            {
                                break;
                            }
                        }
                    }

                    float entropy = 0f;
                    for (int j = 0; j < vocabulary; j++)
                    {
                        if (weights[j] > 0f)
                        {
                            float p = weights[j] / sum;
                            entropy -= p * (CpuMath.Log(p) * 1.44269504f);
                        }
                    }

                    into.Set(r, chosen);
                    sv[o] = chosen;
                    sv[o + 1] = weights[chosen] / sum;
                    sv[o + 2] = entropy;
                    for (int a = 0; a < 5; a++)
                    {
                        int best = -1;
                        for (int j = 0; j < vocabulary; j++)
                        {
                            if (weights[j] > 0f && (best < 0 || weights[j] > weights[best]) && taken[..a].IndexOf(j) < 0)
                            {
                                best = j;
                            }
                        }

                        taken[a] = best;
                        sv[o + 3 + 2 * a] = best;
                        sv[o + 4 + 2 * a] = best < 0 ? 0f : weights[best] / sum;
                    }
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(e);
                ArrayPool<float>.Shared.Return(top);
                foreach (var rented in new[] { candidateScores, candidateWeights })
                {
                    if (rented is not null)
                    {
                        ArrayPool<float>.Shared.Return(rented);
                    }
                }
            }
        }
    }

    public override void PenalizeRowsKernel(Storage logits, Storage work, Storage history, Storage length, int rows, int vocabulary,
        int rowStride, int rowOffset, int capacity, int lastN, float repeat, float presence, float frequency)
    {
        var wz = D(logits);
        var tz = new TypedRanges(logits, 0, "PenalizeRows");                  // a typed row read through its type when copied
        var w = V(work);
        fixed (float* ph = D(history), pl = D(length))
        {
            var ring = IndicesOf(history, ph);
            long count = IndicesOf(length, pl)[0];
            int window = (int)Math.Clamp(Math.Min(count, lastN), 0, capacity);
            int[] recent = ArrayPool<int>.Shared.Rent(Math.Max(1, window));
            try
            {
                for (int r = 0; r < rows; r++)
                {
                    var row = w.AsSpan(r * vocabulary, vocabulary);
                    long first = (long)r * rowStride + rowOffset;
                    var z = tz.Values(wz, first, vocabulary, out long shift);
                    z.AsSpan((int)(shift + first), vocabulary).CopyTo(row);
                    int taken = 0;
                    for (long p = count - window; p < count; p++)
                    {
                        long token = ring[(long)r * capacity + p % capacity];
                        if (token >= 0 && token < vocabulary)
                        {
                            recent[taken++] = (int)token;
                        }
                    }

                    TokenSampler.Penalize(row, recent.AsSpan(0, taken), repeat, presence, frequency);
                }
            }
            finally
            {
                ArrayPool<int>.Shared.Return(recent);
            }
        }
    }

    public override void HistoryPushKernel(Storage ids, Storage history, Storage length, int rows, int capacity)
    {
        fixed (float* pi = D(ids), ph = D(history), pl = D(length))
        {
            var from = IndicesOf(ids, pi);
            var ring = IndicesOf(history, ph);
            long at = IndicesOf(length, pl)[0] % Math.Max(1, capacity);
            for (int r = 0; r < rows; r++)
            {
                ring.Set((long)r * capacity + at, from[r]);
            }
        }
    }

    public override void LogSoftmaxTopKKernel(Storage x, Storage values, Storage indices, int rows, int cols, int k)
    {
        var wx = D(x);
        var tx = new TypedRanges(x, 0, "LogSoftmaxTopK");                     // a typed row read through its type when ranked
        var pv = V(values);
        fixed (float* pi = D(indices))
        {
            var into = IndicesOf(indices, pi);
            for (int r = 0; r < rows; r++)
            {
                var px = tx.Values(wx, (long)r * cols, cols, out long offset);
                var row = px.AsSpan((int)(offset + (long)r * cols), cols);
                var best = pv.AsSpan(r * k, k);
                best.Fill(float.NegativeInfinity);
                for (int j = 0; j < k; j++)
                {
                    into.Set((long)r * k + j, -1);
                }

                // shift = max + ln Σ e^(x - max) in double (the max ignoring NaN, the sum over the entries that are not -∞).
                double max = double.NegativeInfinity;
                bool bad = false;
                foreach (float v in row)
                {
                    bad |= float.IsNaN(v) || float.IsPositiveInfinity(v);
                    if (v > max)
                    {
                        max = v;
                    }
                }

                if (bad || double.IsNegativeInfinity(max))
                {
                    continue;
                }

                double sum = 0;
                foreach (float v in row)
                {
                    if (!float.IsNegativeInfinity(v))
                    {
                        sum += CpuMath.ExpDouble(v - max);
                    }
                }

                float shift = (float)(max + CpuMath.LogDouble(sum));
                int filled = 0;
                for (int j = 0; j < cols; j++)
                {
                    float v = row[j] - shift;
                    if (float.IsNaN(v) || float.IsNegativeInfinity(v) || filled == k && !(v > best[k - 1]))
                    {
                        continue;
                    }

                    int at = filled < k ? filled++ : k - 1;
                    while (at > 0 && best[at - 1] < v)
                    {
                        best[at] = best[at - 1];
                        into.Set((long)r * k + at, into[(long)r * k + at - 1]);
                        at--;
                    }

                    best[at] = v;
                    into.Set((long)r * k + at, j);
                }
            }
        }
    }

    public override void SoftmaxCrossEntropyRowsKernel(Storage logits, Storage targets, Storage weights, Storage losses, int rows, int vocabulary, float scale)
    {
        fixed (float* wz = D(logits), pt = D(targets), ww = D(weights), pl = V(losses))
        {
            // The logits are overwritten with the gradient: typed logits read and written back through their type a chunk of
            // rows at a time (TypedInOut, rows of whole groups of its blocks); a typed weights vector read once, its own size.
            var tw = new TypedInput(weights, ww, 0, rows, "SoftmaxCrossEntropyRows");
            float* pw = tw.Range(0, rows);
            var ids = IndicesOf(targets, pt);
            if (NativeFloats(logits))
            {
                Rows("SoftmaxCrossEntropyRows", new CrossEntropyRows(wz, ids, pw, pl, vocabulary, scale, rows), rows, vocabulary, 24);
                return;
            }

            int step = RowStep(rows, vocabulary, true);
            var tz = new TypedInOut(logits, wz, 0, (long)step * vocabulary, "SoftmaxCrossEntropyRows");
            for (int r = 0; r < rows; r += step)
            {
                int e = Math.Min(rows, r + step);
                long at = (long)r * vocabulary, count = (long)(e - r) * vocabulary;
                Rows("SoftmaxCrossEntropyRows", new CrossEntropyRows(tz.Range(at, count), ids, pw, pl, vocabulary, scale, rows, r, e), e, vocabulary, 24, r);
                tz.Store(at, count);
            }
        }
    }

    // losses[r] = w (lse - x[t]); the logits overwritten with scale · w · (softmax - onehot(t)).
    private struct CrossEntropyRows(float* logits, Indices targets, float* weights, float* losses, int vocabulary, float scale, int rows, int first = 0, int last = -1)
        : IRowKernel
    {
        private float* _z = logits, _losses = losses;

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public readonly void Row<TV, TF>(int r) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            float* row = _z + (long)r * vocabulary;
            long t = targets[r];
            float w = weights[r];
            float max = MaxOf<TV, TF>(row, vocabulary);
            float picked = t >= 0 && t < vocabulary ? row[t] : 0f;
            var exp = new ExpShiftMap(row, row, max);
            RowMap<TV, TF, ExpShiftMap>(ref exp, vocabulary);
            var terms = new SumTerms(row);
            float sum = Reduce16<TV, TF, SumTerms>(ref terms, vocabulary);
            _losses[r] = w == 0f ? 0f : w * (max + Fn<float, F1>.Log(sum) - picked);
            var gradient = new ScaleMap(row, row, scale * w / sum);
            RowMap<TV, TF, ScaleMap>(ref gradient, vocabulary);
            if (t >= 0 && t < vocabulary)
            {
                row[t] -= scale * w;
            }
        }

        public void Redirect(ref Scratch scratch)
        {
            // The logits are read and written in place: measured on a copy of the rows run (a chunk's, rows [first, last)).
            int end = last < 0 ? rows : last;
            long from = (long)first * vocabulary, count = (long)(end - first) * vocabulary;
            float* copy = scratch.Take((long)rows * vocabulary);
            new ReadOnlySpan<float>(_z + from, (int)count).CopyTo(new Span<float>(copy + from, (int)count));
            _z = copy;
            _losses = scratch.Take(rows);
        }
    }
}
