// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;

namespace Idrak.Abstraction.Devices.Cpu;

// Attention over a range of keys per query row (Backend.AttentionSpans), tiled: a block of query rows of one head goes
// through the keys its rows see a tile at a time, the tile's scores a small matrix product (CpuMatMul, one thread per
// block), with an online softmax per row, so no more than a block × tile of scores exists at once. The gradient runs in
// two passes that recompute the weights: one per key/value head and block of keys (dkeys, dvalues), one per query head
// and block of rows (dq), so no two workers add into the same gradient.
internal sealed partial class CpuBackend
{
    // Query rows per block and keys per tile: a 64 × 128 tile of scores (32 KB) and the block's outputs stay in cache.
    private const int SpanRows = 64, SpanKeys = 128;

    private static readonly ParallelOptions OneThread = new() { MaxDegreeOfParallelism = 1 };

    // A product inside a worker: c = a · op(b) (+ c when beta is 1), on the calling thread.
    private static void SpanProduct(float[] a, int aOffset, float[] b, int bOffset, float[] c, int cOffset, int m, int n, int k, bool transA, bool transB,
        float beta) =>
        CpuMatMul.Multiply(a, aOffset, b, bOffset, c, cOffset, m, n, k, transA, transB, beta, long.MaxValue, OneThread);

    // Row i's range of keys, clamped to [0, keyRows] (empty when end ≤ start).
    private static (int Start, int End) SpanOf(float[] starts, float[] ends, int index, int keyRows) =>
        (Math.Clamp((int)starts[index], 0, keyRows), Math.Clamp((int)ends[index], 0, keyRows));

    // The keys a block of rows sees together: from its smallest start to its largest end (empty when no row sees any).
    private static (int Start, int End) BlockSpan(float[] starts, float[] ends, int first, int count, int keyRows)
    {
        int lo = keyRows, hi = 0;
        for (int i = first; i < first + count; i++)
        {
            var (s, e) = SpanOf(starts, ends, i, keyRows);
            if (e > s)
            {
                lo = Math.Min(lo, s);
                hi = Math.Max(hi, e);
            }
        }

        return hi > lo ? (lo, hi) : (0, 0);
    }

    public override void AttentionSpansKernel(Storage q, Storage keys, Storage values, Storage starts, Storage ends, Storage y, Storage? logSumExp, int heads,
        int kvHeads, int headsPerTable, int rows, int keyRows, int dim, float scale, AttentionVariant variant = default)
    {
        float[] qv = D(q), kv = D(keys), vv = D(values), sv = D(starts), ev = D(ends), yv = D(y);
        float[]? lv = logSumExp is null ? null : D(logSumExp);
        int group = heads / kvHeads, blocks = (rows + SpanRows - 1) / SpanRows;
        For(heads * blocks, (long)heads * rows * Math.Max(1, keyRows) * dim, (first, last) =>
        {
            var scores = ArrayPool<float>.Shared.Rent(SpanRows * SpanKeys);
            var outputs = ArrayPool<float>.Shared.Rent(SpanRows * dim);
            Span<float> max = stackalloc float[SpanRows], total = stackalloc float[SpanRows];
            for (int item = first; item < last; item++)
            {
                int h = item / blocks, r0 = item % blocks * SpanRows, count = Math.Min(SpanRows, rows - r0);
                int table = h / headsPerTable * rows, keyBase = h / group * keyRows;
                var (from, to) = BlockSpan(sv, ev, table + r0, count, keyRows);
                max.Fill(float.NegativeInfinity);
                total.Clear();
                Array.Clear(outputs, 0, count * dim);
                for (int c0 = from; c0 < to; c0 += SpanKeys)
                {
                    int width = Math.Min(SpanKeys, to - c0);
                    SpanProduct(qv, (h * rows + r0) * dim, kv, (keyBase + c0) * dim, scores, 0, count, width, dim, false, true, 0f);
                    for (int i = 0; i < count; i++)
                    {
                        var (s, e) = SpanOf(sv, ev, table + r0 + i, keyRows);
                        int a = Math.Max(s, c0) - c0, b = Math.Min(e, c0 + width) - c0;
                        var row = scores.AsSpan(i * width, width);
                        if (b <= a)
                        {
                            row.Clear();                                          // nothing seen here: weights 0, sums unchanged
                            continue;
                        }

                        row[..a].Clear();
                        row[b..].Clear();
                        var seen = row[a..b];
                        float tileMax = float.NegativeInfinity;
                        for (int c = 0; c < seen.Length; c++)
                        {
                            seen[c] = variant.Cap(seen[c] * scale);
                            tileMax = MathF.Max(tileMax, seen[c]);
                        }

                        float updated = MathF.Max(max[i], tileMax), alpha = MathF.Exp(max[i] - updated);
                        total[i] = total[i] * alpha + CpuMath.ExpShifted(seen, updated);
                        max[i] = updated;
                        if (alpha != 1f)
                        {
                            CpuMath.Scale(outputs.AsSpan(i * dim, dim), alpha);
                        }
                    }

                    SpanProduct(scores, 0, vv, (keyBase + c0) * dim, outputs, 0, count, dim, width, false, false, 1f);
                }

                for (int i = 0; i < count; i++)
                {
                    int row = h * rows + r0 + i;
                    var output = yv.AsSpan(row * dim, dim);
                    if (total[i] > 0f)
                    {
                        float inverse = 1f / total[i];
                        for (int d = 0; d < dim; d++)
                        {
                            output[d] = outputs[i * dim + d] * inverse;
                        }
                    }
                    else
                    {
                        output.Clear();                                           // an empty range: zeros
                    }

                    if (lv is not null)
                    {
                        lv[row] = total[i] > 0f ? max[i] + MathF.Log(total[i]) : float.NegativeInfinity;
                    }
                }
            }

            ArrayPool<float>.Shared.Return(outputs);
            ArrayPool<float>.Shared.Return(scores);
        });
    }

    public override void AttentionSpansBackwardKernel(Storage q, Storage keys, Storage values, Storage starts, Storage ends, Storage output, Storage logSumExp,
        Storage dOutput, Storage dq, Storage dkeys, Storage dvalues, int heads, int kvHeads, int headsPerTable, int rows, int keyRows, int dim, float scale,
        AttentionVariant variant = default)
    {
        float[] qv = D(q), kv = D(keys), vv = D(values), sv = D(starts), ev = D(ends), ov = D(output), lv = D(logSumExp), gv = D(dOutput);
        float[] dqv = D(dq), dkv = D(dkeys), dvv = D(dvalues);
        int group = heads / kvHeads, rowBlocks = (rows + SpanRows - 1) / SpanRows, keyBlocks = (keyRows + SpanKeys - 1) / SpanKeys;
        long work = (long)heads * rows * Math.Max(1, keyRows) * dim;

        // Δ per row: dOutput · output.
        var delta = new float[heads * rows];
        For(heads * rows, (long)heads * rows * dim, (first, last) =>
        {
            for (int row = first; row < last; row++)
            {
                delta[row] = Dot(gv.AsSpan(row * dim, dim), ov.AsSpan(row * dim, dim));
            }
        });

        // The keys each block of rows of each table sees (tables × row blocks).
        int tables = heads / headsPerTable;
        var blockSpans = new (int Start, int End)[tables * rowBlocks];
        for (int t = 0; t < tables; t++)
        {
            for (int rb = 0; rb < rowBlocks; rb++)
            {
                int r0 = rb * SpanRows;
                blockSpans[t * rowBlocks + rb] = BlockSpan(sv, ev, t * rows + r0, Math.Min(SpanRows, rows - r0), keyRows);
            }
        }

        // dS for (head h, rows r0.., keys c0..) into `ds` [count, width], scaled by `scale`; P into `p`. False when no
        // row of the block sees any of the keys.
        bool Gradients(int h, int r0, int count, int c0, int width, float[] p, float[] ds)
        {
            int table = h / headsPerTable * rows, keyBase = h / group * keyRows;
            SpanProduct(qv, (h * rows + r0) * dim, kv, (keyBase + c0) * dim, p, 0, count, width, dim, false, true, 0f);
            SpanProduct(gv, (h * rows + r0) * dim, vv, (keyBase + c0) * dim, ds, 0, count, width, dim, false, true, 0f);
            bool any = false;
            for (int i = 0; i < count; i++)
            {
                int row = h * rows + r0 + i;
                var (s, e) = SpanOf(sv, ev, table + r0 + i, keyRows);
                int a = Math.Max(s, c0) - c0, b = Math.Min(e, c0 + width) - c0;
                var pr = p.AsSpan(i * width, width);
                var dr = ds.AsSpan(i * width, width);
                for (int c = 0; c < width; c++)
                {
                    if (c < a || c >= b)
                    {
                        pr[c] = 0f;
                        dr[c] = 0f;
                        continue;
                    }

                    float score = variant.Cap(pr[c] * scale), weight = MathF.Exp(score - lv[row]);
                    pr[c] = weight;
                    dr[c] = scale * weight * (dr[c] - delta[row]) * variant.Slope(score);
                }

                any |= b > a;
            }

            return any;
        }

        // dkeys and dvalues: a worker per key/value head and block of keys, through every query head of its group.
        For(kvHeads * keyBlocks, work, (first, last) =>
        {
            var p = ArrayPool<float>.Shared.Rent(SpanRows * SpanKeys);
            var ds = ArrayPool<float>.Shared.Rent(SpanRows * SpanKeys);
            for (int item = first; item < last; item++)
            {
                int g = item / keyBlocks, c0 = item % keyBlocks * SpanKeys, width = Math.Min(SpanKeys, keyRows - c0);
                for (int h = g * group; h < (g + 1) * group; h++)
                {
                    for (int rb = 0; rb < rowBlocks; rb++)
                    {
                        var (from, to) = blockSpans[h / headsPerTable * rowBlocks + rb];
                        int r0 = rb * SpanRows, count = Math.Min(SpanRows, rows - r0);
                        if (to <= c0 || from >= c0 + width || !Gradients(h, r0, count, c0, width, p, ds))
                        {
                            continue;
                        }

                        SpanProduct(p, 0, gv, (h * rows + r0) * dim, dvv, (g * keyRows + c0) * dim, width, dim, count, true, false, 1f);
                        SpanProduct(ds, 0, qv, (h * rows + r0) * dim, dkv, (g * keyRows + c0) * dim, width, dim, count, true, false, 1f);
                    }
                }
            }

            ArrayPool<float>.Shared.Return(ds);
            ArrayPool<float>.Shared.Return(p);
        });

        // dq: a worker per query head and block of rows, through the keys its rows see.
        For(heads * rowBlocks, work, (first, last) =>
        {
            var p = ArrayPool<float>.Shared.Rent(SpanRows * SpanKeys);
            var ds = ArrayPool<float>.Shared.Rent(SpanRows * SpanKeys);
            for (int item = first; item < last; item++)
            {
                int h = item / rowBlocks, rb = item % rowBlocks, r0 = rb * SpanRows, count = Math.Min(SpanRows, rows - r0);
                var (from, to) = blockSpans[h / headsPerTable * rowBlocks + rb];
                for (int c0 = from; c0 < to; c0 += SpanKeys)
                {
                    int width = Math.Min(SpanKeys, to - c0);
                    if (Gradients(h, r0, count, c0, width, p, ds))
                    {
                        SpanProduct(ds, 0, kv, (h / group * keyRows + c0) * dim, dqv, (h * rows + r0) * dim, count, dim, width, false, false, 1f);
                    }
                }
            }

            ArrayPool<float>.Shared.Return(ds);
            ArrayPool<float>.Shared.Return(p);
        });
    }
}
