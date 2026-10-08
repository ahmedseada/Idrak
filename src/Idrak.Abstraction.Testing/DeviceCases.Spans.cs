// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;

namespace Idrak.Abstraction.Testing;

// Attention over one range of keys per query row (Backend.AttentionSpans and its gradient) for every rule KeySpans
// builds and for ranges drawn at random (empty ones and ones past the keys among them), grouped heads and several
// tables, against plain loops that follow the contract in Backend.cs.
public static partial class DeviceCases
{
    // Backend.AttentionSpans by plain loops in double: y and the log-sum-exp (-∞ and zeros for an empty range).
    private static (float[] Y, float[] LogSumExp) SpanAttentionReference(float[] q, float[] k, float[] v, float[] starts, float[] ends, int heads, int kvHeads,
        int headsPerTable, int rows, int keyRows, int dim, float scale, AttentionVariant variant)
    {
        var y = new float[heads * rows * dim];
        var lse = new float[heads * rows];
        var scores = new double[keyRows];
        for (int h = 0; h < heads; h++)
        {
            int g = h / (heads / kvHeads), table = h / headsPerTable;
            for (int i = 0; i < rows; i++)
            {
                int row = h * rows + i;
                int first = Math.Clamp((int)starts[table * rows + i], 0, keyRows), end = Math.Clamp((int)ends[table * rows + i], 0, keyRows);
                if (end <= first)
                {
                    lse[row] = float.NegativeInfinity;
                    continue;
                }

                double max = double.NegativeInfinity;
                for (int c = first; c < end; c++)
                {
                    double dot = 0;
                    for (int d = 0; d < dim; d++)
                    {
                        dot += (double)q[row * dim + d] * k[(g * keyRows + c) * dim + d];
                    }

                    scores[c] = variant.Softcap > 0f ? variant.Softcap * Math.Tanh(scale * dot / variant.Softcap) : scale * dot;
                    max = Math.Max(max, scores[c]);
                }

                double sum = 0;
                for (int c = first; c < end; c++)
                {
                    scores[c] = Math.Exp(scores[c] - max);
                    sum += scores[c];
                }

                for (int d = 0; d < dim; d++)
                {
                    double acc = 0;
                    for (int c = first; c < end; c++)
                    {
                        acc += scores[c] * v[(g * keyRows + c) * dim + d];
                    }

                    y[row * dim + d] = (float)(acc / sum);
                }

                lse[row] = (float)(max + Math.Log(sum));
            }
        }

        return (y, lse);
    }

    // Backend.AttentionSpansBackward by plain loops in double: the gradients added to copies of dq, dk and dv.
    private static (float[] Dq, float[] Dk, float[] Dv) SpanAttentionBackwardReference(float[] q, float[] k, float[] v, float[] starts, float[] ends,
        float[] y, float[] lse, float[] dy, float[] dq, float[] dk, float[] dv, int heads, int kvHeads, int headsPerTable, int rows, int keyRows, int dim,
        float scale, AttentionVariant variant)
    {
        double[] dqa = new double[dq.Length], dka = new double[dk.Length], dva = new double[dv.Length];
        for (int h = 0; h < heads; h++)
        {
            int g = h / (heads / kvHeads), table = h / headsPerTable;
            for (int i = 0; i < rows; i++)
            {
                int row = h * rows + i;
                int first = Math.Clamp((int)starts[table * rows + i], 0, keyRows), end = Math.Clamp((int)ends[table * rows + i], 0, keyRows);
                double delta = 0;
                for (int d = 0; d < dim; d++)
                {
                    delta += (double)dy[row * dim + d] * y[row * dim + d];
                }

                for (int c = first; c < end; c++)
                {
                    int key = (g * keyRows + c) * dim;
                    double dot = 0, dp = 0;
                    for (int d = 0; d < dim; d++)
                    {
                        dot += (double)q[row * dim + d] * k[key + d];
                        dp += (double)dy[row * dim + d] * v[key + d];
                    }

                    double score = scale * dot, slope = 1;
                    if (variant.Softcap > 0f)
                    {
                        score = variant.Softcap * Math.Tanh(score / variant.Softcap);
                        slope = 1 - score / variant.Softcap * (score / variant.Softcap);
                    }

                    double p = Math.Exp(score - lse[row]), ds = p * (dp - delta) * slope;
                    for (int d = 0; d < dim; d++)
                    {
                        dqa[row * dim + d] += scale * ds * k[key + d];
                        dka[key + d] += scale * ds * q[row * dim + d];
                        dva[key + d] += p * dy[row * dim + d];
                    }
                }
            }
        }

        static float[] Added(float[] values, double[] added) => [.. values.Select((x, i) => (float)(x + added[i]))];
        return (Added(dq, dqa), Added(dk, dka), Added(dv, dva));
    }

    // The ranges of one table of `rows` rows over `keyRows` keys, by rule.
    private static KeySpans SpanRule(DeviceCaseContext c, string rule, int rows, int keyRows) => rule switch
    {
        "bidirectional" => KeySpans.Bidirectional(rows, keyRows),
        "causal" => KeySpans.Causal(rows),
        "window" => KeySpans.Causal(rows, c.Random.Next(1, rows + 2)),
        "segments" => KeySpans.Segments(SplitInto(c, rows), c.Random.Next(2) == 0 ? 0 : c.Random.Next(1, 9)),
        "image blocks" => KeySpans.ImageBlocks(rows, ImageBlockRuns(c, rows), c.Random.Next(2) == 0 ? 0 : c.Random.Next(1, rows + 2)),
        _ => new KeySpans(                                                     // at random: empty ranges and ends past the keys too
            [.. Enumerable.Range(0, rows).Select(_ => c.Random.Next(0, keyRows + 2))],
            [.. Enumerable.Range(0, rows).Select(_ => c.Random.Next(0, keyRows + 3))]),
    };

    // Lengths adding up to `rows` (packed sequences).
    private static List<int> SplitInto(DeviceCaseContext c, int rows)
    {
        var lengths = new List<int>();
        for (int left = rows; left > 0;)
        {
            int length = Math.Min(left, c.Random.Next(1, Math.Max(2, rows / 2 + 1)));
            lengths.Add(length);
            left -= length;
        }

        return lengths;
    }

    // Image blocks that do not overlap, with text between them.
    private static List<(int Start, int Length)> ImageBlockRuns(DeviceCaseContext c, int rows)
    {
        var blocks = new List<(int, int)>();
        for (int at = c.Random.Next(0, 3); at < rows;)
        {
            int length = Math.Min(rows - at, c.Random.Next(1, 17));
            blocks.Add((at, length));
            at += length + c.Random.Next(1, 6);
        }

        return blocks;
    }

    // Every rule, odd sizes (rows past one block of 64, keys past one tile of 128), grouped heads, one table per head
    // group or one for all, with and without a soft-cap; the forward against plain loops, the gradients too.
    private static void SpanAttention(DeviceCaseContext c)
    {
        var b = c.Backend;
        foreach (string rule in new[] { "bidirectional", "causal", "window", "segments", "image blocks", "random" })
        {
            int kvHeads = c.Size(1, 2), group = c.Random.Next(1, 4), heads = kvHeads * group, tables = c.Random.Next(2) == 0 ? 1 : heads;
            int rows = c.Size(1, 150), keyRows = rule is "bidirectional" or "random" ? c.Size(1, 150) : rows, dim = c.Size(1, 80);
            int headsPerTable = heads / tables;
            float scale = 1f / MathF.Sqrt(dim);
            var variant = new AttentionVariant(0, c.Random.Next(3) == 0 ? 2f + c.Random.NextSingle() * 10f : 0f);
            var spans = KeySpans.Concat([.. Enumerable.Range(0, tables).Select(_ => SpanRule(c, rule, rows, keyRows))]);
            float[] sv = [.. spans.Starts.Select(s => (float)s)], ev = [.. spans.Ends.Select(e => (float)e)];
            float[] qv = c.Values(heads * rows * dim, 2f), kv = c.Values(kvHeads * keyRows * dim), vv = c.Values(kvHeads * keyRows * dim);
            Storage q = c.Storage(qv), k = c.Storage(kv), v = c.Storage(vv), starts = c.Storage(sv), ends = c.Storage(ev);
            Storage y = Random(c, heads * rows * dim), lse = Random(c, heads * rows);
            string what = $"attention over {rule} ranges ({heads} heads, {kvHeads} key/value heads, {tables} tables, {rows} rows, {keyRows} keys, dim {dim})";

            var (expectedY, expectedLse) = SpanAttentionReference(qv, kv, vv, sv, ev, heads, kvHeads, headsPerTable, rows, keyRows, dim, scale, variant);
            b.AttentionSpans(q, k, v, starts, ends, y, lse, heads, kvHeads, headsPerTable, rows, keyRows, dim, scale, variant);
            c.ExpectClose(expectedY, Read(y), 1e-4f, what);
            c.ExpectClose(expectedLse, Read(lse), 1e-4f, $"{what}: the log-sum-exp");
            b.AttentionSpans(q, k, v, starts, ends, Random(c, heads * rows * dim), null, heads, kvHeads, headsPerTable, rows, keyRows, dim, scale, variant);

            float[] dyv = c.Values(heads * rows * dim), dqv = c.Values(heads * rows * dim), dkv = c.Values(kvHeads * keyRows * dim), dvv = c.Values(kvHeads * keyRows * dim);
            Storage dq = c.Storage(dqv), dk = c.Storage(dkv), dv = c.Storage(dvv);
            var (eq, ek, evv) = SpanAttentionBackwardReference(qv, kv, vv, sv, ev, Read(y), Read(lse), dyv, dqv, dkv, dvv, heads, kvHeads, headsPerTable, rows,
                keyRows, dim, scale, variant);
            b.AttentionSpansBackward(q, k, v, starts, ends, y, lse, c.Storage(dyv), dq, dk, dv, heads, kvHeads, headsPerTable, rows, keyRows, dim, scale, variant);
            c.ExpectClose(eq, Read(dq), 1e-3f, $"{what}: dq");
            c.ExpectClose(ek, Read(dk), 1e-3f, $"{what}: dkeys");
            c.ExpectClose(evv, Read(dv), 1e-3f, $"{what}: dvalues");
        }

        // The rules themselves, on a small sequence whose ranges are known (inclusive last keys in the comments).
        c.Expect(KeySpans.ImageBlocks(8, [(2, 3)]).Ends.SequenceEqual([1, 2, 5, 5, 5, 6, 7, 8]), "image blocks: rows 2 to 4 see keys up to 4");
        c.Expect(KeySpans.ImageBlocks(8, [(2, 3)], window: 2).Starts.SequenceEqual([0, 0, 1, 2, 3, 4, 5, 6]), "image blocks in a window of 2 start at i - 1");
        c.Expect(KeySpans.Segments([3, 2]).Starts.SequenceEqual([0, 0, 0, 3, 3]), "segments start at their sequence");
    }
}
