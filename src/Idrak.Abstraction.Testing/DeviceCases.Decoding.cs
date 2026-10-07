// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Operations;

namespace Idrak.Abstraction.Testing;

// The cases of packed weights, gated activations, key/value caches, attention, sampling and autograd.
public static partial class DeviceCases
{
    private static readonly PackedFormat[] Formats = [PackedFormat.Int8, PackedFormat.Int4, PackedFormat.BFloat16];

    // y = x · w for packed weights, on the CPU (the reference of the fused products).
    private static void PackedProduct(Backend cpu, PackedWeight weight, Storage x, Storage y, int m)
    {
        int k = weight.Rows, n = weight.Columns;
        switch (weight.Format)
        {
            case PackedFormat.Int8:
                cpu.Int8MatMul(x, weight.PackedValues.Storage, weight.ScaleValues!.Storage, y, m, n, k);
                break;
            case PackedFormat.Int4:
                cpu.Int4MatMul(x, weight.PackedValues.Storage, weight.ScaleValues!.Storage, y, m, n, k);
                break;
            default:
                cpu.BFloat16MatMul(x, weight.PackedValues.Storage, y, m, n, k);
                break;
        }
    }

    private static void PackedWeights(DeviceCaseContext c)
    {
        var b = c.Backend;
        int k = c.Size(1, 96), n = c.Size(1, 80);
        var w = c.Values(k * n, 0.5f);
        foreach (var format in Formats)
        {
            using var weight = PackedWeight.FromValues(format, w, k, n, c.Device);
            var scales = weight.ScaleValues?.Storage;
            foreach (int m in new[] { 1, c.Size(2, 8), c.Size(9, 40) })
            {
                PackedProduct(b, weight, Random(c, m * k), c.Zeros(m * n), m);
            }

            switch (format)
            {
                case PackedFormat.Int8:
                    b.Int8Dequantize(weight.PackedValues.Storage, scales!, c.Zeros(k * n), k, n);
                    break;
                case PackedFormat.Int4:
                    b.Int4Dequantize(weight.PackedValues.Storage, scales!, c.Zeros(k * n), k, n);
                    break;
                default:
                    b.BFloat16Dequantize(weight.PackedValues.Storage, c.Zeros(k * n), k, n);
                    break;
            }
        }

        foreach (int n2 in new[] { 1, 2, 7, c.Size(2, 300) })
        {
            var values = c.Values(n2, 10f);
            values[0] = 1.00390625f;                                              // a tie: rounds to even
            b.PackBFloat16(c.Storage(values), c.Zeros((n2 + 1) / 2), n2);
        }
    }

    private static void FusedPackedProducts(DeviceCaseContext c)
    {
        var b = c.Backend;
        int k = 8 * c.Size(1, 12), n = 2 * c.Size(1, 40);
        foreach (var format in Formats)
        {
            using var weight = PackedWeight.FromValues(format, c.Values(k * n, 0.5f), k, n, c.Device);
            Storage packed = weight.PackedValues.Storage;
            Storage? scales = weight.ScaleValues?.Storage;

            // Many rows (a prompt).
            int rows = c.Size(16, 48);
            Storage x = Random(c, rows * k), y = c.Zeros(rows * n);
            c.Composed(Ops.PackedMatMulLarge, cpu => cpu.PackedMatMulLarge(format, x, packed, scales, y, rows, n, k), cpu => PackedProduct(cpu, weight, x, y, rows));

            // Few rows: the down projection after the gate, and the projection with the residual and the next norm.
            foreach (int m in new[] { 1, c.Size(2, 4) })
            {
                foreach (int activation in new[] { 0, 1 })
                {
                    Storage gate = Random(c, m * k), up = Random(c, m * k), down = c.Zeros(m * n);
                    c.Composed(Ops.PackedMatMulGated, cpu => cpu.PackedMatMulGated(format, activation, gate, up, packed, scales, down, m, n, k), cpu =>
                    {
                        var hidden = c.Zeros(m * k);
                        cpu.GatedActivation(gate, up, hidden, m * k, activation);
                        PackedProduct(cpu, weight, hidden, down, m);
                    });
                }

                Storage input = Random(c, m * k), output = c.Zeros(m * n), residual = Random(c, m * n), sum = c.Zeros(m * n), gain = Random(c, n), normalized = c.Zeros(m * n);
                float offset = c.Random.Next(2);
                c.Composed(Ops.PackedMatMulAddRmsNorm,
                    cpu => cpu.PackedMatMulAddRmsNorm(format, input, packed, scales, output, m, n, k, residual, sum, gain, normalized, 1e-6f, offset), cpu =>
                    {
                        PackedProduct(cpu, weight, input, output, m);
                        cpu.AddRmsNormAffine(residual, output, sum, gain, normalized, m, n, 1e-6f, offset);
                    });

                // Several products of one input, then a gate and up pair with its activation.
                using var second = PackedWeight.FromValues(format, c.Values(k * n, 0.5f), k, n, c.Device);
                PackedWeight[] weights = [weight, second];
                var products = weights.Select((p, i) => (Packed: p.PackedValues.Storage, Scales: p.ScaleValues?.Storage, Bias: i == 0 ? Random(c, n) : null, Output: c.Zeros(m * n), Columns: n)).ToArray();
                c.Composed(Ops.PackedMatMulMany, cpu => cpu.PackedMatMulMany(format, input, m, k, products), cpu =>
                {
                    for (int i = 0; i < products.Length; i++)
                    {
                        PackedProduct(cpu, weights[i], input, products[i].Output, m);
                        if (products[i].Bias is { } bias)
                        {
                            cpu.AddRowVector(products[i].Output, bias, products[i].Output, m, n);
                        }
                    }
                });

                foreach (int activation in new[] { 0, 1, 2 })
                {
                    var pair = weights.Select(p => (Packed: p.PackedValues.Storage, Scales: p.ScaleValues?.Storage, Bias: (Storage?)null, Output: c.Zeros(m * n), Columns: n)).ToArray();
                    var hidden = c.Zeros(m * n);
                    c.Composed(Ops.PackedMatMulGatedPair, cpu => cpu.PackedMatMulGatedPair(format, activation, input, m, k, pair, hidden), cpu =>
                    {
                        PackedProduct(cpu, weights[0], input, pair[0].Output, m);
                        PackedProduct(cpu, weights[1], input, pair[1].Output, m);
                        cpu.GatedActivation(pair[0].Output, pair[1].Output, hidden, m * n, activation);
                    });
                }

                // Each product with a low-rank term (an adapter): y_j = x · w_j + u_j · v_j.
                int rank = c.Size(1, 16);
                var lowRank = weights.Select(p => (Packed: p.PackedValues.Storage, Scales: p.ScaleValues?.Storage, Output: c.Zeros(m * n), Columns: n,
                    U: Random(c, m * rank), V: Random(c, rank * n))).ToArray();
                c.Composed(Ops.PackedMatMulLowRank, cpu => cpu.PackedMatMulLowRank(format, input, m, k, lowRank, rank), cpu =>
                {
                    for (int i = 0; i < lowRank.Length; i++)
                    {
                        PackedProduct(cpu, weights[i], input, lowRank[i].Output, m);
                        cpu.MatMul(lowRank[i].U, lowRank[i].V, lowRank[i].Output, m, n, rank, false, false, 1f);
                    }
                });
            }

            // The input gradient through a frozen bfloat16 weight [k, n] read as stored: dx [m, k] = beta·dx + g [m, n] · wᵀ (+ u · vᵀ).
            if (format == PackedFormat.BFloat16)
            {
                int m = c.Size(1, 24), rank = c.Size(1, 8);
                foreach (bool adapter in new[] { false, true })
                {
                    float beta = c.Random.Next(2);
                    Storage g = Random(c, m * n), dx = Random(c, m * k), u = Random(c, m * rank), v = Random(c, k * rank);
                    c.Composed(Ops.BFloat16TransposedMatMul, cpu => cpu.BFloat16TransposedMatMul(g, packed, dx, m, k, n, beta, adapter ? u : null, adapter ? v : null, adapter ? rank : 0), cpu =>
                    {
                        var expanded = c.Zeros(k * n);
                        cpu.BFloat16Dequantize(packed, expanded, k, n);
                        cpu.MatMul(g, expanded, dx, m, k, n, false, true, beta);
                        if (adapter)
                        {
                            cpu.MatMul(u, v, dx, m, k, rank, false, true, 1f);
                        }
                    });
                }
            }
        }
    }

    private static void GatedActivations(DeviceCaseContext c)
    {
        var b = c.Backend;
        foreach (int n in Sizes(c, 1, 2, 37))
        {
            int words = (n + 1) / 2;
            float[] gv = c.Values(n, 3f), uv = c.Values(n, 3f);
            Storage gate = c.Storage(gv), up = c.Storage(uv), packedGate = c.Storage(PackRows(gv, 1, n)), packedUp = c.Storage(PackRows(uv, 1, n));
            foreach (int kind in new[] { 0, 1, 2 })
            {
                b.GatedActivation(gate, up, c.Zeros(n), n, kind);
                foreach (int flags in new[] { 3, 1, 2, 15 })
                {
                    b.GatedActivationBackward(gate, up, Random(c, n), Random(c, n), Random(c, n), n, kind, flags);
                    b.GatedActivationBackwardPacked(packedGate, packedUp, Random(c, n), Random(c, n), Random(c, n), n, kind, flags);
                }

                // Read floats or packed words; write y, the packed inputs, the packed y.
                foreach (int flags in new[] { 4, 4 | 2, 4 | 8, 1 | 4, 1 | 8, 2 | 4 | 8 })
                {
                    b.GatedActivationPacked(gate, up, (flags & 1) != 0 ? packedGate : c.Zeros(words), (flags & 1) != 0 ? packedUp : c.Zeros(words),
                        c.Zeros(n), c.Zeros(words), n, kind, flags);
                }
            }
        }
    }

    // ------------------------------------------------------------------ decoding

    private static void KeyValueCaches(DeviceCaseContext c)
    {
        var b = c.Backend;
        int heads = c.Size(1, 4), steps = c.Size(1, 6), dim = c.Size(1, 64), capacity = steps + c.Size(1, 20), start = c.Random.Next(capacity - steps + 1);
        var position = c.Storage([start]);
        b.DecoderMask(position, c.Zeros(steps * capacity), steps, capacity);
        b.KeyValueWrite(Random(c, heads * steps * dim), Random(c, heads * capacity * dim), position, heads, steps, capacity, dim);
        int int8Words = (dim + 3) / 4, halfWords = (dim + 1) / 2;
        Storage cache = c.Zeros(heads * capacity * int8Words), scales = c.Zeros(heads * capacity);
        b.KeyValueWriteInt8(Random(c, heads * steps * dim, 2f), cache, scales, position, heads, steps, capacity, dim);
        b.KeyValueWriteBFloat16(Random(c, heads * steps * dim), c.Zeros(heads * capacity * halfWords), position, heads, steps, capacity, dim);

        // Scores and context over a whole int8 cache.
        b.KeyValueWriteInt8(Random(c, heads * capacity * dim, 2f), cache, scales, c.Storage([0f]), heads, capacity, capacity, dim);
        b.AttentionScoresInt8(Random(c, heads * steps * dim), cache, scales, c.Zeros(heads * steps * capacity), heads, steps, capacity, dim);
        b.AttentionContextInt8(Random(c, heads * steps * capacity), cache, scales, c.Zeros(heads * steps * dim), heads, steps, capacity, dim);
    }

    private static readonly AttentionVariant[] Variants = [default, new(3, 0f), new(0, 30f), new(4, 20f)];

    private static void Attention(DeviceCaseContext c)
    {
        var b = c.Backend;
        int heads = c.Size(1, 4), group = c.Random.Next(1, 3), steps = c.Size(1, 7), dim = 8 * c.Size(1, 8), capacity = steps + c.Size(1, 24);
        int rows = group * steps, start = c.Random.Next(capacity - steps + 1);
        float scale = 1f / MathF.Sqrt(dim);
        var position = c.Storage([start]);
        Storage keys = Random(c, heads * capacity * dim), values = Random(c, heads * capacity * dim);
        int int8Words = (dim + 3) / 4, halfWords = (dim + 1) / 2;
        Storage keys8 = c.Zeros(heads * capacity * int8Words), values8 = c.Zeros(heads * capacity * int8Words), keyScales = c.Zeros(heads * capacity), valueScales = c.Zeros(heads * capacity);
        Storage keys16 = c.Zeros(heads * capacity * halfWords), values16 = c.Zeros(heads * capacity * halfWords), zero = c.Storage([0f]);
        b.KeyValueWriteInt8(Random(c, heads * capacity * dim), keys8, keyScales, zero, heads, capacity, capacity, dim);
        b.KeyValueWriteInt8(Random(c, heads * capacity * dim), values8, valueScales, zero, heads, capacity, capacity, dim);
        b.KeyValueWriteBFloat16(Random(c, heads * capacity * dim), keys16, zero, heads, capacity, capacity, dim);
        b.KeyValueWriteBFloat16(Random(c, heads * capacity * dim), values16, zero, heads, capacity, capacity, dim);
        foreach (var variant in Variants)
        {
            var q = Random(c, heads * rows * dim, 2f);
            b.AttentionDecode(q, keys, values, position, c.Zeros(heads * rows * dim), heads, rows, steps, capacity, dim, scale, variant);
            b.AttentionTiled(q, keys, values, position, c.Zeros(heads * rows * dim), null, heads, rows, steps, capacity, dim, scale, variant);
            foreach (bool tiled in new[] { false, true })
            {
                b.AttentionInt8(q, keys8, values8, keyScales, valueScales, position, c.Zeros(heads * rows * dim), heads, rows, steps, capacity, dim, scale, tiled, variant);
                b.AttentionBFloat16(q, keys16, values16, position, c.Zeros(heads * rows * dim), heads, rows, steps, capacity, dim, scale, tiled, variant);
            }

            // Training: the whole sequence from position 0, its log-sum-exp and the gradients.
            Storage tk = Random(c, heads * steps * dim), tv = Random(c, heads * steps * dim), y = c.Zeros(heads * rows * dim), lse = c.Zeros(heads * rows);
            b.AttentionTiled(q, tk, tv, zero, y, lse, heads, rows, steps, steps, dim, scale, variant);
            b.AttentionTiledBackward(q, tk, tv, y, lse, Random(c, heads * rows * dim), Random(c, heads * rows * dim), Random(c, heads * steps * dim),
                Random(c, heads * steps * dim), heads, rows, steps, steps, dim, scale, variant);

            // Packed sequences: each packed row of `steps` positions split into sequences.
            int headsPerRow = heads, packedRows = heads / headsPerRow;
            var starts = new float[packedRows * steps];
            var ends = new float[packedRows * steps];
            for (int r = 0; r < packedRows; r++)
            {
                for (int t = 0, first = 0; t < steps; t++)
                {
                    if (t > 0 && c.Random.Next(3) == 0)
                    {
                        first = t;
                    }

                    starts[r * steps + t] = first;
                }

                for (int t = steps - 1, end = steps; t >= 0; t--)
                {
                    ends[r * steps + t] = end;
                    if (starts[r * steps + t] == t)
                    {
                        end = t;
                    }
                }
            }

            Storage st = c.Storage(starts), en = c.Storage(ends), sy = c.Zeros(heads * rows * dim), slse = c.Zeros(heads * rows);
            b.AttentionSegmented(q, tk, tv, sy, slse, st, en, heads, headsPerRow, rows, steps, dim, scale, variant);
            b.AttentionSegmented(q, tk, tv, c.Zeros(heads * rows * dim), null, st, en, heads, headsPerRow, rows, steps, dim, scale, variant);
            b.AttentionSegmentedBackward(q, tk, tv, sy, slse, Random(c, heads * rows * dim), Random(c, heads * rows * dim), Random(c, heads * steps * dim),
                Random(c, heads * steps * dim), st, en, heads, headsPerRow, rows, steps, dim, scale, variant);

            // Rows of different lengths decoded together (one head per row here; a row may be all padding).
            var rowStarts = c.Storage([.. Enumerable.Range(0, heads * steps).Select(i => (float)c.Random.Next(0, start + i % steps + 2))]);
            b.AttentionRows(q, keys, values, position, c.Zeros(heads * rows * dim), rowStarts, heads, 1, rows, steps, capacity, dim, scale, variant);
        }

        NormRopeHeads(c);
    }

    // The attention layer's heads normalized, rotated and laid out (and written into a cache), against plain loops.
    private static void NormRopeHeads(DeviceCaseContext c)
    {
        int batch = c.Size(1, 2), steps = c.Size(1, 5), kvHeads = c.Size(1, 3), group = c.Random.Next(1, 3), heads = kvHeads * group, cols = 2 * c.Size(2, 32);
        int half = c.Random.Next(2) == 0 ? cols / 2 : cols / 4, maxPositions = 32;
        var (cos, sin) = RotaryTables(c, maxPositions, Math.Max(half, 1));
        float[] qv = c.Values(batch * steps * heads * cols, 2f), kv = c.Values(batch * steps * kvHeads * cols, 2f), vv = c.Values(batch * steps * kvHeads * cols);
        float[] gq = c.Values(cols), gk = c.Values(cols), pv = [.. Enumerable.Range(0, steps).Select(t => (float)((t * 5 + 1) % maxPositions))];
        float[] cosValues = Read(cos), sinValues = Read(sin);
        foreach (bool normalize in new[] { false, true })
        {
            foreach (bool rotate in new[] { false, true })
            {
                foreach (bool cached in new[] { false, true })
                {
                    foreach (bool bfloat16 in cached ? new[] { false, true } : [false])
                    {
                        bool interleaved = c.Random.Next(2) == 0;
                        int capacity = cached ? steps + c.Size(1, 8) : steps, start = cached ? c.Random.Next(capacity - steps + 1) : 0;
                        int stride = cols + (bfloat16 ? 2 : 0) * c.Random.Next(2), words = bfloat16 ? stride / 2 : stride;
                        Storage q = c.Storage(qv), k = c.Storage(kv), v = c.Storage(vv), positions = c.Storage(pv);
                        Storage yq = c.Zeros(batch * heads * steps * cols), yk = c.Zeros(batch * kvHeads * capacity * words), yv = c.Zeros(batch * kvHeads * capacity * words);
                        Storage? position = cached ? c.Storage([start]) : null;
                        c.Composed(Ops.NormRopeHeads, cpu => cpu.NormRopeHeads(q, k, v, batch, steps, heads, kvHeads, cols, normalize ? c.Storage(gq) : null, 1e-6f, 1f,
                            normalize ? c.Storage(gk) : null, 1e-5f, 0f, rotate ? cos : null, rotate ? sin : null, positions, rotate ? half : 0, interleaved,
                            yq, yk, yv, position, capacity, stride, bfloat16), cpu =>
                        {
                            // One head's vector: normalized with gain, then rotated.
                            float[] Head(float[] source, int offset, float[]? gain, float eps, float shift, int t)
                            {
                                var x = source.AsSpan(offset, cols).ToArray();
                                if (gain is not null)
                                {
                                    double squares = 0;
                                    foreach (float value in x)
                                    {
                                        squares += value * value;
                                    }

                                    float inv = 1f / MathF.Sqrt((float)(squares / cols) + eps);
                                    for (int d = 0; d < cols; d++)
                                    {
                                        x[d] = x[d] * inv * (gain[d] + shift);
                                    }
                                }

                                if (rotate)
                                {
                                    int p = (int)pv[t];
                                    var r = (float[])x.Clone();
                                    for (int i = 0; i < half; i++)
                                    {
                                        int a = interleaved ? 2 * i : i, bIndex = interleaved ? 2 * i + 1 : i + half;
                                        float co = cosValues[p * half + i], si = sinValues[p * half + i];
                                        r[a] = x[a] * co - x[bIndex] * si;
                                        r[bIndex] = x[bIndex] * co + x[a] * si;
                                    }

                                    x = r;
                                }

                                return x;
                            }

                            float[] oq = Read(yq), ok = Read(yk), ov = Read(yv);
                            void Write(float[] target, int row, float[] x)
                            {
                                if (!bfloat16)
                                {
                                    x.CopyTo(target, row * words);
                                    return;
                                }

                                var packedRow = PackRows(x, 1, cols);
                                packedRow.CopyTo(target, row * words);
                            }

                            for (int n = 0; n < batch; n++)
                            {
                                for (int t = 0; t < steps; t++)
                                {
                                    for (int h = 0; h < heads; h++)
                                    {
                                        var x = Head(qv, ((n * steps + t) * heads + h) * cols, normalize ? gq : null, 1e-6f, 1f, t);
                                        x.CopyTo(oq, ((n * heads + h) * steps + t) * cols);
                                    }

                                    for (int h = 0; h < kvHeads; h++)
                                    {
                                        int row = (n * kvHeads + h) * capacity + start + t;
                                        Write(ok, row, Head(kv, ((n * steps + t) * kvHeads + h) * cols, normalize ? gk : null, 1e-5f, 0f, t));
                                        Write(ov, row, vv.AsSpan(((n * steps + t) * kvHeads + h) * cols, cols).ToArray());
                                    }
                                }
                            }

                            cpu.Upload(oq, yq);
                            cpu.Upload(ok, yk);
                            cpu.Upload(ov, yv);
                        });
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------ sampling and autograd

    // The sampler's operations as a generation step runs them: penalties from the token history, a draw per row (from
    // the last of several positions too), and the history extended.
    private static void Sampling(DeviceCaseContext c)
    {
        var b = c.Backend;
        int rows = c.Size(1, 4), vocabulary = c.Size(2, 1000), steps = c.Size(1, 4), positions = c.Random.Next(1, 3), capacity = 32;
        Storage history = Indices(c, rows * capacity, vocabulary), length = c.Storage([c.Random.Next(0, 40)]);
        Storage ids = c.Zeros(rows), stats = c.Zeros(steps * rows * 13);
        foreach (var (topK, topP, minP, penalties) in new[] { (1, 1f, 0f, false), (0, 0.9f, 0f, true), (40, 1f, 0.05f, false), (5, 0.5f, 0f, true) })
        {
            uint seed = (uint)c.Random.Next();
            for (int s = 0; s < steps; s++)
            {
                Storage source = Random(c, rows * positions * vocabulary, 5f), step = c.Storage([s]);
                int rowStride = positions * vocabulary, rowOffset = (positions - 1) * vocabulary;
                if (penalties)
                {
                    var work = c.Zeros(rows * vocabulary);
                    b.PenalizeRows(source, work, history, length, rows, vocabulary, rowStride, rowOffset, capacity, 16, 1.3f, 0.4f, 0.2f);
                    (source, rowStride, rowOffset) = (work, vocabulary, 0);
                }

                b.SampleRows(source, ids, stats, step, rows, vocabulary, rowStride, rowOffset, 0.8f, topK, topP, minP, seed);
                b.HistoryPush(ids, history, length, rows, capacity);
            }
        }
    }

    // A small network (two layers, a residual, a softmax head) forward and backward through tensors, as training runs it.
    private static void Autograd(DeviceCaseContext c)
    {
        int batch = c.Size(1, 16), inputs = c.Size(1, 24), hidden = c.Size(2, 32), classes = c.Size(2, 10);
        using var w1 = Tensor.From(c.Values(inputs * hidden, 0.5f), [inputs, hidden], c.Device, requiresGrad: true);
        using var b1 = Tensor.From(c.Values(hidden, 0.1f), [hidden], c.Device, requiresGrad: true);
        using var w2 = Tensor.From(c.Values(hidden * classes, 0.5f), [hidden, classes], c.Device, requiresGrad: true);
        using var x = Tensor.From(c.Values(batch * inputs), [batch, inputs], c.Device, requiresGrad: true);
        using var targets = Tensor.From([.. Enumerable.Range(0, batch * classes).Select(i => i % classes == i / classes % classes ? 1f : 0f)], [batch, classes], c.Device);
        var h = (x.MatMul(w1) + b1).Gelu();
        var g = h.Sigmoid() * h.Tanh() + (h.Square() + 1f).Sqrt();
        var logits = g.MatMul(w2);
        var loss = (logits.LogSoftmax() * targets).Sum() * (-1f / batch) + logits.Softmax().Square().Mean() + x.Abs().Mean() + h.Transpose().Exp().Mean();
        loss.Backward();
        c.Expect(float.IsFinite(loss.Item()) && w1.Grad is not null && x.Grad is not null, "the loss and gradients are finite");
    }
}
