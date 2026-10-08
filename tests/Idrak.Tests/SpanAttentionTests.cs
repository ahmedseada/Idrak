// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Abstraction.Operations;
using Idrak.Layers;

// Attention over one range of keys per query row (Tensor.AttentionSpans, Backend.AttentionSpans): every rule KeySpans
// builds against the composed path (the full scores, a mask, softmax), forward and gradients, on each device; the
// rules' ranges as Gemma 3 masks them; and the bidirectional multi-head attention layer, which now runs on it.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] SpanAttentionGroup =
    [
        ("attention spans: bidirectional, causal, window, segment and image-block ranges (two tables, grouped heads, a soft-cap) match the composed scores, forward and gradients; empty ranges give zeros and no gradient", SpanAttentionMatchesComposed),
        ("attention spans: the rules' ranges are Gemma 3's masks (a key seen when not after the row or in the row's image block, within the window on sliding layers)", SpanRulesAreGemmaMasks),
        ("attention spans: bidirectional multi-head attention runs on the key-range kernel (no full score matrix) and matches the composed path, forward and gradients", BidirectionalAttentionLayer),
    ];

    // The composed path for query head h: softmax(cap(scale · q kᵀ) + mask) v with the mask from the table's ranges.
    private static Tensor ComposedSpans(Tensor q, Tensor k, Tensor v, KeySpans spans, int tables, float scale, AttentionVariant variant)
    {
        int heads = q.Shape[0], rows = q.Shape[1], kvHeads = k.Shape[0], keyRows = k.Shape[1];
        var outputs = new List<Tensor>();
        for (int h = 0; h < heads; h++)
        {
            int g = h / (heads / kvHeads), table = h / (heads / tables);
            var mask = new float[rows * keyRows];
            for (int i = 0; i < rows; i++)
            {
                for (int c = 0; c < keyRows; c++)
                {
                    mask[i * keyRows + c] = c >= spans.Starts[table * rows + i] && c < spans.Ends[table * rows + i] ? 0f : -1e9f;
                }
            }

            var scores = q.Narrow(0, h, 1).MatMul(k.Narrow(0, g, 1), transposeB: true) * scale;
            if (variant.Softcap > 0f)
            {
                scores = (scores * (1f / variant.Softcap)).Tanh() * variant.Softcap;
            }

            outputs.Add((scores + Tensor.From(mask, [rows, keyRows], q.Device)).Softmax().MatMul(v.Narrow(0, g, 1)));
        }

        return Tensor.Concat(outputs, 0);
    }

    private static void SpanAttentionMatchesComposed(Device device)
    {
        var random = new Random(71);
        int rows = 37, dim = 13;
        float scale = 1f / MathF.Sqrt(dim);
        (string Name, Func<KeySpans> Spans)[] rules =
        [
            ("bidirectional", () => KeySpans.Bidirectional(rows, rows)),
            ("causal", () => KeySpans.Causal(rows)),
            ("window", () => KeySpans.Causal(rows, 5)),
            ("segments", () => KeySpans.Segments([10, 1, 20, 6], window: 7)),
            ("image blocks", () => KeySpans.ImageBlocks(rows, [(3, 9), (20, 16)])),
            ("image blocks in a window", () => KeySpans.ImageBlocks(rows, [(3, 9), (20, 16)], window: 4)),
        ];
        foreach (var (name, make) in rules)
        {
            foreach (var variant in new[] { default(AttentionVariant), new AttentionVariant(0, 3f) })
            {
                int kvHeads = 2, heads = 6, tables = 2;
                var spans = KeySpans.Concat(make(), name == "bidirectional" ? KeySpans.Causal(rows) : make());
                var (starts, ends) = spans.ToTensors(device);
                var w = Tensor.From(RandomArray(random, heads * rows * dim), [heads, rows, dim], device);
                Tensor Leaf(int count, int[] shape) => Tensor.From(RandomArray(random, count), shape, device, requiresGrad: true);
                var (q, k, v) = (Leaf(heads * rows * dim, [heads, rows, dim]), Leaf(kvHeads * rows * dim, [kvHeads, rows, dim]), Leaf(kvHeads * rows * dim, [kvHeads, rows, dim]));
                var y = Tensor.AttentionSpans(q, k, v, starts, ends, scale, variant);
                (y * w).Sum().Backward();
                float[] gq = q.Grad!.ToArray(), gk = k.Grad!.ToArray(), gv = v.Grad!.ToArray();
                foreach (var t in new[] { q, k, v })
                {
                    t.ZeroGrad();
                }

                var expected = ComposedSpans(q, k, v, spans, tables, scale, variant);
                (expected * w).Sum().Backward();
                string what = $"{name}, cap {variant.Softcap} on {device}";
                AssertClose(expected.ToArray(), y.ToArray(), 2e-4f, what);
                AssertClose(q.Grad!.ToArray(), gq, 1e-3f, what + ": dq");
                AssertClose(k.Grad!.ToArray(), gk, 1e-3f, what + ": dkeys");
                AssertClose(v.Grad!.ToArray(), gv, 1e-3f, what + ": dvalues");
            }
        }

        // Empty ranges (rows 0 and 2) and ranges past the keys (row 1, clamped): zeros, and no gradient from those rows.
        var empty = new KeySpans([3, 0, 4, 0], [3, 9, 1, 2]);
        var (es, ee) = empty.ToTensors(device);
        var eq = Tensor.From(RandomArray(random, 4 * 3), [1, 4, 3], device, requiresGrad: true);
        var ek = Tensor.From(RandomArray(random, 5 * 3), [1, 5, 3], device, requiresGrad: true);
        var ey = Tensor.AttentionSpans(eq, ek, ek, es, ee, 0.5f);
        var values = ey.ToArray();
        Check(values[..3].All(x => x == 0f) && values[6..9].All(x => x == 0f) && values.All(float.IsFinite), "empty ranges give zeros");
        ey.Sum().Backward();
        var dq = eq.Grad!.ToArray();
        Check(dq[..3].All(x => x == 0f) && dq[6..9].All(x => x == 0f) && dq.All(float.IsFinite), "empty ranges give no gradient");
    }

    private static void SpanRulesAreGemmaMasks(Device device)
    {
        int rows = 23;
        (int Start, int Length)[] blocks = [(2, 6), (12, 4), (19, 4)];
        foreach (int window in new[] { 0, 1, 3, 8, 40 })
        {
            var spans = KeySpans.ImageBlocks(rows, blocks, window);
            for (int q = 0; q < rows; q++)
            {
                int block = Array.FindIndex(blocks, b => q >= b.Start && q < b.Start + b.Length);
                for (int k = 0; k < rows; k++)
                {
                    bool sameBlock = block >= 0 && k >= blocks[block].Start && k < blocks[block].Start + blocks[block].Length;
                    bool seen = (k <= q || sameBlock) && (window == 0 || k > q - window);
                    bool ranged = k >= spans.Starts[q] && k < spans.Ends[q];
                    Check(seen == ranged, $"window {window}: row {q}, key {k} is {(seen ? "" : "not ")}seen by Gemma 3's mask");
                }
            }
        }
    }

    private static void BidirectionalAttentionLayer(Device device)
    {
        using var attention = new MultiHeadAttention(16, 4, device: device, random: new Random(72));
        var x = Tensor.From(RandomArray(new Random(73), 2 * 9 * 16), [2, 9, 16], device, requiresGrad: true);
        var y = attention.Forward(x);

        // The same layer composed by hand from its weights: the full scores, softmax, values.
        var weights = attention.Parameters().ToArray();                          // qkv weight, bias, output weight, bias
        var qkv = x.MatMul(weights[0]) + weights[1];
        Tensor Heads(int part) => qkv.Narrow(2, part * 16, 16).Reshape(2, 9, 4, 4).Permute(0, 2, 1, 3).Reshape(8, 9, 4);
        var context = (Heads(0).MatMul(Heads(1), transposeB: true) * 0.5f).Softmax().MatMul(Heads(2)).Reshape(2, 4, 9, 4).Permute(0, 2, 1, 3).Reshape(2, 9, 16);
        var expected = context.MatMul(weights[2]) + weights[3];
        AssertClose(expected.ToArray(), y.ToArray(), 2e-4f, $"bidirectional attention on {device}");

        y.Square().Sum().Backward();
        var grads = attention.Parameters().Select(p => p.Grad!.ToArray()).Append(x.Grad!.ToArray()).ToArray();
        foreach (var p in attention.Parameters())
        {
            p.ZeroGrad();
        }

        x.ZeroGrad();
        expected.Square().Sum().Backward();
        var reference = attention.Parameters().Select(p => p.Grad!.ToArray()).Append(x.Grad!.ToArray()).ToArray();
        for (int i = 0; i < grads.Length; i++)
        {
            AssertClose(reference[i], grads[i], 1e-3f, $"bidirectional attention gradient {i} on {device}");
        }

        // No [T, T] scores: the layer calls the key-range kernel, never a softmax over scores.
        using (Autograd.NoGrad())
        using (var trace = Kernels.Trace(device.Backend))
        {
            attention.Forward(x);
            Check(trace.Calls(Ops.AttentionSpans) == 1 && trace.Calls(Ops.ScaleMaskSoftmax) == 0 && trace.Calls(Ops.Softmax) == 0,
                "the layer runs AttentionSpans, not a softmax over full scores");
        }
    }
}
