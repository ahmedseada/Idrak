// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;

namespace Idrak.Abstraction.Testing;

// The gradient of attention over one range of keys per query row (Backend.AttentionSpansBackward) at the sizes the
// devices' kernels tile: query rows past several blocks, keys past several tiles, head sizes from 1 to 256 (odd ones,
// 64, 72, 80, 128, and past 128, where a device may take another path), every rule a training step meets (random
// ranges with empty ones and ends past the keys, bidirectional, causal, sliding windows, image blocks in a window),
// grouped heads, one table for all heads or one per head, a soft-cap. The CPU's gradients are checked against plain
// loops, and the gradients Tensor.AttentionSpans records against the composed path's (the full scores, a mask, softmax
// and the products, through autograd); the kit then replays every call on the device.
public static partial class DeviceCases
{
    // Head sizes the gradient is drawn at.
    private static readonly int[] SpanGradientDims = [1, 5, 13, 32, 64, 72, 80, 100, 128, 136, 200, 256];

    private static void SpanAttentionGradient(DeviceCaseContext c)
    {
        var b = c.Backend;
        foreach (string rule in new[] { "random", "bidirectional", "causal", "window", "image blocks" })
        {
            int dim = SpanGradientDims[c.Random.Next(SpanGradientDims.Length)];
            bool wide = dim > 128;                                             // fewer rows and keys: the plain loops stay quick
            int kvHeads = c.Size(1, 2), group = c.Random.Next(1, 4), heads = kvHeads * group, tables = c.Random.Next(2) == 0 ? 1 : heads;
            int rows = wide ? c.Size(17, 70) : c.Size(33, 160);
            int keyRows = rule is "bidirectional" or "random" ? (wide ? c.Size(1, 90) : c.Size(1, 200)) : rows;
            int headsPerTable = heads / tables;
            float scale = 1f / MathF.Sqrt(dim);
            var variant = new AttentionVariant(0, c.Random.Next(3) == 0 ? 2f + c.Random.NextSingle() * 10f : 0f);
            var spans = KeySpans.Concat([.. Enumerable.Range(0, tables).Select(_ => SpanRule(c, rule, rows, keyRows))]);
            float[] sv = [.. spans.Starts.Select(s => (float)s)], ev = [.. spans.Ends.Select(e => (float)e)];
            float[] qv = c.Values(heads * rows * dim, 2f), kv = c.Values(kvHeads * keyRows * dim), vv = c.Values(kvHeads * keyRows * dim);
            Storage q = c.Storage(qv), k = c.Storage(kv), v = c.Storage(vv), starts = c.Storage(sv), ends = c.Storage(ev);
            Storage y = c.Zeros(heads * rows * dim), lse = c.Zeros(heads * rows);
            string what = $"the gradient of attention over {rule} ranges ({heads} heads, {kvHeads} key/value heads, {tables} tables, {rows} rows, "
                + $"{keyRows} keys, dim {dim}, cap {variant.Softcap:G3})";

            b.AttentionSpans(q, k, v, starts, ends, y, lse, heads, kvHeads, headsPerTable, rows, keyRows, dim, scale, variant);
            float[] dyv = c.Values(heads * rows * dim), dqv = c.Values(heads * rows * dim), dkv = c.Values(kvHeads * keyRows * dim), dvv = c.Values(kvHeads * keyRows * dim);
            Storage dq = c.Storage(dqv), dk = c.Storage(dkv), dv = c.Storage(dvv);
            var (eq, ek, evv) = SpanAttentionBackwardReference(qv, kv, vv, sv, ev, Read(y), Read(lse), dyv, dqv, dkv, dvv, heads, kvHeads, headsPerTable, rows,
                keyRows, dim, scale, variant);
            b.AttentionSpansBackward(q, k, v, starts, ends, y, lse, c.Storage(dyv), dq, dk, dv, heads, kvHeads, headsPerTable, rows, keyRows, dim, scale, variant);
            c.ExpectClose(eq, Read(dq), 1e-3f, $"{what}: dq");
            c.ExpectClose(ek, Read(dk), 1e-3f, $"{what}: dkeys");
            c.ExpectClose(evv, Read(dv), 1e-3f, $"{what}: dvalues");
        }

        SpanGradientThroughTensors(c);
    }

    // Tensor.AttentionSpans' gradients against the composed path's, through autograd, for a rule whose every row sees a
    // key (where a row sees none, the composed softmax would spread its weight over the masked keys).
    private static void SpanGradientThroughTensors(DeviceCaseContext c)
    {
        string rule = new[] { "bidirectional", "causal", "window", "image blocks" }[c.Random.Next(4)];
        int dim = new[] { 8, 13, 64, 72 }[c.Random.Next(4)];
        int kvHeads = c.Size(1, 2), group = c.Random.Next(1, 3), heads = kvHeads * group, tables = c.Random.Next(2) == 0 ? 1 : heads;
        int rows = c.Size(2, 70), keyRows = rule == "bidirectional" ? c.Size(1, 70) : rows;
        float scale = 1f / MathF.Sqrt(dim);
        var variant = new AttentionVariant(0, c.Random.Next(2) == 0 ? 3f : 0f);
        var spans = KeySpans.Concat([.. Enumerable.Range(0, tables).Select(_ => SpanRule(c, rule, rows, keyRows))]);
        var (starts, ends) = spans.ToTensors(c.Device);
        using var weights = Tensor.From(c.Values(heads * rows * dim), [heads, rows, dim], c.Device);
        using var q = Tensor.From(c.Values(heads * rows * dim), [heads, rows, dim], c.Device, requiresGrad: true);
        using var k = Tensor.From(c.Values(kvHeads * keyRows * dim), [kvHeads, keyRows, dim], c.Device, requiresGrad: true);
        using var v = Tensor.From(c.Values(kvHeads * keyRows * dim), [kvHeads, keyRows, dim], c.Device, requiresGrad: true);
        string what = $"the recorded gradient over {rule} ranges ({heads} heads, {kvHeads} key/value heads, {tables} tables, {rows} rows, {keyRows} keys, "
            + $"dim {dim}, cap {variant.Softcap:G3}) against the composed path's";

        (Tensor.AttentionSpans(q, k, v, starts, ends, scale, variant) * weights).Sum().Backward();
        float[] gq = q.Grad!.ToArray(), gk = k.Grad!.ToArray(), gv = v.Grad!.ToArray();
        foreach (var t in new[] { q, k, v })
        {
            t.ZeroGrad();
        }

        // The composed path, a query head at a time: softmax(cap(scale · q kᵀ) + mask) v, the mask from the head's table.
        var outputs = new List<Tensor>();
        for (int h = 0; h < heads; h++)
        {
            int g = h / group, table = h / (heads / tables);
            var mask = new float[rows * keyRows];
            for (int i = 0; i < rows; i++)
            {
                for (int key = 0; key < keyRows; key++)
                {
                    bool seen = key >= spans.Starts[table * rows + i] && key < spans.Ends[table * rows + i];
                    mask[i * keyRows + key] = seen ? 0f : -1e9f;
                }
            }

            var scores = q.Narrow(0, h, 1).MatMul(k.Narrow(0, g, 1), transposeB: true) * scale;
            if (variant.Softcap > 0f)
            {
                scores = (scores * (1f / variant.Softcap)).Tanh() * variant.Softcap;
            }

            outputs.Add((scores + Tensor.From(mask, [rows, keyRows], c.Device)).Softmax().MatMul(v.Narrow(0, g, 1)));
        }

        (Tensor.Concat(outputs, 0) * weights).Sum().Backward();
        c.ExpectClose(q.Grad!.ToArray(), gq, 1e-3f, $"{what}: dq");
        c.ExpectClose(k.Grad!.ToArray(), gk, 1e-3f, $"{what}: dkeys");
        c.ExpectClose(v.Grad!.ToArray(), gv, 1e-3f, $"{what}: dvalues");
    }
}
