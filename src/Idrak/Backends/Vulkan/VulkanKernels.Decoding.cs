// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Backends.Vulkan;

// Kernels for language models: rotary positions, embedding gathers, the key/value cache, the decoder mask, and attention
// over the cache for decoding. Positions and indices are float values in device memory, read and truncated as the CPU.
internal static partial class VulkanKernels
{
    private static IEnumerable<(string, Func<SpirvKernel>)> DecodingKernels()
    {
        // Rotary embedding of x [rows = batch·steps·heads, dim] into y (already holding x): pair p of row r at step
        // t = r / heads % steps turns by cos/sin[positions[t], p] (sin · sign). Pairs (2p, 2p + 1) when interleaved ≠ 0,
        // else (p, p + half). One invocation per pair.
        yield return ("rope", () =>
        {
            var k = new KernelBuilder("rope", Block);
            var (x, y, cos, sin, positions) = (k.Buffer("x"), k.Buffer("y"), k.Buffer("cos"), k.Buffer("sin"), k.Buffer("positions"));
            var (rows, heads, steps, dim, half) = (k.PushInt("rows"), k.PushInt("heads"), k.PushInt("steps"), k.PushInt("dim"), k.PushInt("half"));
            var (interleaved, sign) = (k.PushInt("interleaved"), k.PushFloat("sign"));
            Grid(k, rows * half, idx =>
            {
                var (r, p) = (idx / half, idx % half);
                var position = positions[r / heads % steps].ToInt();
                var c = cos[position * half + p];
                var s = sin[position * half + p] * sign;
                var o = r * dim;
                var pairs = interleaved.Ne(0);
                var i = o + k.Select(pairs, 2 * p, p);
                var j = o + k.Select(pairs, 2 * p + 1, p + half);
                var (va, vb) = (x[i], x[j]);
                y[i] = k.Fma(va, c, -(vb * s));
                y[j] = k.Fma(vb, c, va * s);
            });
            return k.Build();
        });

        // y[i, :] = table[indices[i], :]; indices outside [0, vocabulary) are clamped into it (the CPU throws).
        yield return ("gather", () =>
        {
            var k = new KernelBuilder("gather", Block);
            var (table, indices, y) = (k.Buffer("table"), k.Buffer("indices"), k.Buffer("y"));
            var (count, dim, vocabulary) = (k.PushInt("count"), k.PushInt("dim"), k.PushInt("vocabulary"));
            Grid(k, count * dim, i =>
            {
                var (row, d) = (i / dim, i % dim);
                var index = k.Clamp(indices[row].ToInt(), k.Int(0), vocabulary - 1);
                y[i] = table[index * dim + d];
            });
            return k.Build();
        });

        // gather from a bfloat16 table [vocabulary, dim] (rows of ⌈dim / 2⌉ words, low half first).
        yield return ("gather_bf16", () =>
        {
            var k = new KernelBuilder("gather_bf16", Block);
            var (packed, indices, y) = (k.Buffer("packed"), k.Buffer("indices"), k.Buffer("y"));
            var (count, dim, vocabulary) = (k.PushInt("count"), k.PushInt("dim"), k.PushInt("vocabulary"));
            Grid(k, count * dim, i =>
            {
                var (row, d) = (i / dim, i % dim);
                var index = k.Clamp(indices[row].ToInt(), k.Int(0), vocabulary - 1);
                y[i] = BFloat16At(k, packed, index * ((dim + 1) / 2) + (d >> 1), d);
            });
            return k.Build();
        });

        // cache[h, position + t, :] = source[h, t, :] for [heads, steps, dim] → [heads, capacity, dim].
        yield return ("key_value_write", () =>
        {
            var k = new KernelBuilder("key_value_write", Block);
            var (source, cache, position) = (k.Buffer("source"), k.Buffer("cache"), k.Buffer("position"));
            var (heads, steps, capacity, dim) = (k.PushInt("heads"), k.PushInt("steps"), k.PushInt("capacity"), k.PushInt("dim"));
            var start = position[k.Int(0)].ToInt();
            var block = steps * dim;
            Grid(k, heads * block, i =>
            {
                var h = i / block;
                cache[(h * capacity + start) * dim + i % block] = source[i];
            });
            return k.Build();
        });

        // key_value_write into a bfloat16 cache (rows of ⌈dim / 2⌉ words, rounded to nearest, ties to even). One
        // invocation per cache word.
        yield return ("key_value_write_bf16", () =>
        {
            var k = new KernelBuilder("key_value_write_bf16", Block);
            var (source, cache, position) = (k.Buffer("source"), k.Buffer("cache"), k.Buffer("position"));
            var (heads, steps, capacity, dim) = (k.PushInt("heads"), k.PushInt("steps"), k.PushInt("capacity"), k.PushInt("dim"));
            var start = position[k.Int(0)].ToInt();
            var words = (dim + 1) / 2;
            Grid(k, heads * steps * words, i =>
            {
                var (row, w) = (i / words, i % words);                          // row = h · steps + t
                var (h, t) = (row / steps, row % steps);
                var from = row * dim + 2 * w;
                var low = RoundBFloat16(k, source[from]);
                var high = k.Local(k.UInt(0));
                k.If(2 * w + 1 < dim, () => high.V = RoundBFloat16(k, source[from + 1]));
                cache[(h * capacity + start + t) * words + w] = (low | high.V << 16).AsFloat();
            });
            return k.Build();
        });

        // Quantizes source [heads·steps, dim] into the int8 cache at positions position[0] + step: per row,
        // scale = max|x| / 127 and bytes round(x / scale) (ties to even) in [-127, 127]. One invocation per row.
        yield return ("key_value_write_int8", () =>
        {
            var k = new KernelBuilder("key_value_write_int8", Block);
            var (source, cache, scales, position) = (k.Buffer("source"), k.Buffer("cache"), k.Buffer("scales"), k.Buffer("position"));
            var (heads, steps, capacity, dim) = (k.PushInt("heads"), k.PushInt("steps"), k.PushInt("capacity"), k.PushInt("dim"));
            var start = position[k.Int(0)].ToInt();
            var words = (dim + 3) / 4;
            Grid(k, heads * steps, row =>
            {
                var slot = row / steps * capacity + start + row % steps;
                var o = row * dim;
                var max = k.Local(0f);
                k.For(k.Int(0), dim, 1, d => max.V = k.Max(max.V, k.Abs(source[o + d])));
                var scale = max.V / 127f;
                var inverse = k.Select(max.V > 0f, 1f / scale, k.Float(0f));
                scales[slot] = scale;
                k.For(k.Int(0), words, 1, w =>
                {
                    var bits = k.Local(k.Int(0));
                    for (int b = 0; b < 4; b++)
                    {
                        int lane = b;
                        var d = 4 * w + lane;
                        k.If(d < dim, () =>
                        {
                            var q = k.Clamp(k.RoundEven(source[o + d] * inverse).ToInt(), k.Int(-127), k.Int(127));
                            bits.V = bits.V | (q & 0xFF) << (8 * lane);
                        });
                    }

                    cache[slot * words + w] = bits.V.AsFloat();
                });
            });
            return k.Build();
        });

        // mask[i, j] = j ≤ position + i ? 0 : -1e9 for a [rows, capacity] mask.
        yield return ("decoder_mask", () =>
        {
            var k = new KernelBuilder("decoder_mask", Block);
            var (position, mask) = (k.Buffer("position"), k.Buffer("mask"));
            var (rows, capacity) = (k.PushInt("rows"), k.PushInt("capacity"));
            var start = position[k.Int(0)].ToInt();
            Grid(k, rows * capacity, idx =>
            {
                var (i, j) = (idx / capacity, idx % capacity);
                mask[idx] = k.Select(j <= start + i, k.Float(0f), k.Float(-1e9f));
            });
            return k.Build();
        });

        yield return ("attention_decode", () => Attention("attention_decode", CacheFormat.Float));

        // attention_decode over a bfloat16 cache: keys and values [heads, capacity, ⌈dim / 2⌉ words].
        yield return ("attention_bf16", () => Attention("attention_bf16", CacheFormat.BFloat16));

        // attention_decode over an int8 cache: keys and values [heads, capacity, ⌈dim / 4⌉ words] of signed bytes, one scale
        // per cached row (keyScales, valueScales [heads, capacity]).
        yield return ("attention_int8", () => Attention("attention_int8", CacheFormat.Int8));
    }

    private enum CacheFormat
    {
        Float,
        BFloat16,
        Int8,
    }

    // Attention over a key/value cache [heads, capacity, dim] for query rows q [heads, rowsPerHead, dim]: row i of head h
    // sees positions c = 0 … min(position[0] + i % rowsPerHead % steps, capacity - 1). One workgroup per query row
    // (dim ≤ 256: invocation d keeps output dimension d). Pass 1: each invocation scores its positions for the row's
    // maximum. Pass 2, 256 positions at a time: the invocations score them again into workgroup memory as
    // exp(score - max), then each output dimension adds its weighted values. y = Σ e_c · v_c / Σ e_c.
    // TODO(tuning): split long contexts across workgroups; cooperative dot products (subgroups) instead of one per lane.
    private static SpirvKernel Attention(string name, CacheFormat format)
    {
        var k = new KernelBuilder(name, Block);
        var (q, keys, values) = (k.Buffer("q"), k.Buffer("keys"), k.Buffer("values"));
        var (keyScales, valueScales) = format == CacheFormat.Int8 ? (k.Buffer("keyScales"), k.Buffer("valueScales")) : (null, null);
        var (position, y) = (k.Buffer("position"), k.Buffer("y"));
        var (heads, rowsPerHead, steps, capacity, dim) = (k.PushInt("heads"), k.PushInt("rowsPerHead"), k.PushInt("steps"), k.PushInt("capacity"), k.PushInt("dim"));
        var scale = k.PushFloat("scale");
        var query = k.Shared("query", AttentionMaxDim);
        var weights = k.Shared("weights", Block);
        var scratch = k.Shared("scratch", Block);
        var lane = k.LocalX;
        var start = position[k.Int(0)].ToInt();
        var words = format switch
        {
            CacheFormat.BFloat16 => (dim + 1) / 2,
            CacheFormat.Int8 => (dim + 3) / 4,
            _ => dim,
        };

        // Dimension d of cached row `slot` (= head · capacity + position) of a cache.
        Val At(Buf cache, Val slot, Val d) => format switch
        {
            CacheFormat.BFloat16 => BFloat16At(k, cache, slot * words + (d >> 1), d),
            CacheFormat.Int8 => Int8At(k, cache, slot * words, d),
            _ => cache[slot * dim + d],
        };

        EachRow(k, heads * rowsPerHead, row =>
        {
            var h = row / rowsPerHead;
            var count = k.Min(start + row % rowsPerHead % steps, capacity - 1) + 1;
            var first = h * capacity;
            k.If(lane < dim, () => query[lane] = q[row * dim + lane]);
            k.Barrier();

            Val Score(Val c)
            {
                var dot = k.Local(0f);
                var slot = first + c;
                k.For(k.Int(0), dim, 1, d => dot.V = dot.V + query[d] * At(keys, slot, d));
                return keyScales is null ? dot.V * scale : dot.V * keyScales[slot] * scale;
            }

            var best = k.Local(float.NegativeInfinity);
            k.For(lane, count, Block, c => best.V = k.Max(best.V, Score(c)));
            var max = k.ReduceMax(scratch, best.V);

            var acc = k.Local(0f);
            var total = k.Local(0f);
            k.For(k.Int(0), count, Block, c0 =>
            {
                var c = c0 + lane;
                var e = k.Local(0f);
                var weighted = k.Local(0f);                              // e times the value row's scale (int8)
                k.If(c < count, () =>
                {
                    e.V = k.Exp(Score(c) - max);
                    weighted.V = valueScales is null ? e.V : e.V * valueScales[first + c];
                });
                weights[lane] = weighted.V;
                total.V = total.V + e.V;
                k.Barrier();
                k.If(lane < dim, () =>
                {
                    var chunk = k.Min(count - c0, k.Int(Block));
                    k.For(k.Int(0), chunk, 1, w => acc.V = k.Fma(weights[w], At(values, first + c0 + w, lane), acc.V));
                });
                k.Barrier();
            });
            var sum = k.ReduceSum(scratch, total.V);
            k.If(lane < dim, () => y[row * dim + lane] = acc.V / sum);
        });
        return k.Build();
    }
}
