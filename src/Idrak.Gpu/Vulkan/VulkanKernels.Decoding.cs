// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Vulkan;

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

        // y[i, :] = table[indices[i], :]; indices outside [0, vocabulary) are clamped into it (the CPU throws). The table's
        // binding holds rows [first, first + rows) (the whole table, or one window of a table larger than a binding:
        // VulkanBackend.LargeStorage.cs); outputs whose row is elsewhere are left to the other windows' dispatches.
        yield return ("gather", () =>
        {
            var k = new KernelBuilder("gather", Block);
            var (table, indices, y) = (k.Buffer("table"), k.Buffer("indices"), k.Buffer("y"));
            var (count, dim, vocabulary) = (k.PushInt("count"), k.PushInt("dim"), k.PushInt("vocabulary"));
            var (first, rows) = (k.PushInt("first"), k.PushInt("rows"));
            Grid(k, count * dim, i =>
            {
                var (row, d) = (i / dim, i % dim);
                var index = k.Clamp(indices[row].ToInt(), k.Int(0), vocabulary - 1) - first;
                k.If((index >= 0) & (index < rows), () => y[i] = table[index * dim + d]);
            });
            return k.Build();
        });

        // gather from a bfloat16 table [vocabulary, dim] (rows of ⌈dim / 2⌉ words, low half first).
        yield return ("gather_bf16", () =>
        {
            var k = new KernelBuilder("gather_bf16", Block);
            var (packed, indices, y) = (k.Buffer("packed"), k.Buffer("indices"), k.Buffer("y"));
            var (count, dim, vocabulary) = (k.PushInt("count"), k.PushInt("dim"), k.PushInt("vocabulary"));
            var (first, rows) = (k.PushInt("first"), k.PushInt("rows"));
            Grid(k, count * dim, i =>
            {
                var (row, d) = (i / dim, i % dim);
                var index = k.Clamp(indices[row].ToInt(), k.Int(0), vocabulary - 1) - first;
                k.If((index >= 0) & (index < rows), () => y[i] = BFloat16At(k, packed, index * ((dim + 1) / 2) + (d >> 1), d));
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

        // Merges the splits of a split attention dispatch: part [rows, splits, dim + 2] holds each split's weighted values
        // Σ e_c · v_c (e_c = exp(score_c - its max)), then its max and Σ e_c. y[row, d] = Σ_s a_s · acc_s[d] / Σ_s a_s · sum_s
        // with a_s = exp(max_s - max over s): splits in order; an empty split (max -∞, sums 0) adds nothing. One invocation
        // per output.
        yield return ("attention_combine", () =>
        {
            var k = new KernelBuilder("attention_combine", Block);
            var (part, y) = (k.Buffer("part"), k.Buffer("y"));
            var (rows, splits, dim) = (k.PushInt("rows"), k.PushInt("splits"), k.PushInt("dim"));
            Grid(k, rows * dim, i =>
            {
                var (row, d) = (i / dim, i % dim);
                var stride = dim + 2;
                var first = row * splits * stride;
                var max = k.Local(part[first + dim]);
                k.For(k.Int(1), splits, 1, s => max.V = k.Max(max.V, part[first + s * stride + dim]));
                var (num, den) = (k.Local(0f), k.Local(0f));
                k.For(k.Int(0), splits, 1, s =>
                {
                    var at = first + s * stride;
                    var weight = k.Exp(part[at + dim] - max.V);
                    num.V = k.Fma(weight, part[at + d], num.V);
                    den.V = k.Fma(weight, part[at + dim + 1], den.V);
                });
                y[i] = num.V / den.V;
            });
            return k.Build();
        });
    }

    /// <summary>Fewest positions a split of a split attention dispatch takes, per workgroup width (an eighth of a tile;
    /// shorter contexts use fewer splits).</summary>
    public static int AttentionMinChunk(int width) => width / 8;

    private enum CacheFormat
    {
        Float,
        BFloat16,
        Int8,
    }

    // Attention over a key/value cache [heads, capacity, dim] for query rows q [heads, rowsPerHead, dim]: row i of head h
    // sees positions c = lo … count - 1, count = min(position[0] + i % rowsPerHead % steps, capacity - 1) + 1, lo =
    // max(count - window, 0) with a window (window > 0), else 0.
    // Grid: x = query rows (a group takes rows x, x + groups, …), y = splits: split s takes positions
    // [lo + s · chunk, min(count, lo + (s + 1) · chunk)), chunk = max(⌈(count - lo) / splits⌉, width / 8), computed here
    // from the current length (so the dispatch shape depends only on the shapes). A workgroup of `width` (W) goes through
    // its positions W at a time (a tile) with an online softmax:
    //   1. invocation t scores position c0 + t (q · k_c · scale, soft-capped when softcap > 0, -∞ past the split) into
    //      scratch → the tile's max by a reduction; max' = max(max, tile max), every running sum is rescaled by exp(max - max');
    //   2. invocation t stores e_t = exp(score_t - max') (int8: times the value row's scale) in weights[t] and adds e_t
    //      to its running total;
    //   3. invocation (p, d) = (t >> dimShift, t & (2^dimShift - 1)) adds Σ weights[w] · v[c0 + w, d + j·W] over the
    //      tile's positions w ≡ p (mod W >> dimShift), for j < max(1, 256 / W) (dimShift = log2 of the head size rounded up to a
    //      power of two, at most log2 W: small heads spread the positions over parts of the workgroup, heads wider than
    //      W give each invocation several dimensions).
    // At the end the totals are summed, the parts p of each dimension added in order, and the row is written: y = acc /
    // total with one split, else (acc, max, total) to part[row, split] for attention_combine. A split with no positions
    // writes (0, -∞, 0).
    //
    // Workgroup memory — every phase is separated by a barrier:
    //   query[0 … max(255, W - 1)] written at the start of a row (after the previous row's final barrier), read while scoring;
    //   scratch[W]: the reductions (each ends with a barrier after the result is read), then the parts' sums per j
    //   (written after a barrier ending the previous reads, read after the next);
    //   weights[W]: weights[t] written by invocation t after the tile max's reduction (whose barriers follow the barrier
    //   that ends the previous tile's reads of weights), read after the next barrier.
    private static SpirvKernel Attention(string name, CacheFormat format)
    {
        int lanes = Block, dimsPerLane = Math.Max(1, AttentionMaxDim / lanes);
        var k = new KernelBuilder(name, lanes);
        var (q, keys, values) = (k.Buffer("q"), k.Buffer("keys"), k.Buffer("values"));
        var (keyScales, valueScales) = format == CacheFormat.Int8 ? (k.Buffer("keyScales"), k.Buffer("valueScales")) : (null, null);
        var (position, y) = (k.Buffer("position"), k.Buffer("y"));
        var (heads, rowsPerHead, steps, capacity, dim) = (k.PushInt("heads"), k.PushInt("rowsPerHead"), k.PushInt("steps"), k.PushInt("capacity"), k.PushInt("dim"));
        var (scale, dimShift) = (k.PushFloat("scale"), k.PushInt("dimShift"));
        var (window, softcap) = (k.PushInt("window"), k.PushFloat("softcap"));
        var query = k.Shared("query", dimsPerLane * lanes);
        var weights = k.Shared("weights", lanes);
        var scratch = k.Shared("scratch", lanes);
        var lane = k.LocalX;
        var start = position[k.Int(0)].ToInt();
        var splits = k.GroupsY;
        var words = format switch
        {
            CacheFormat.BFloat16 => (dim + 1) / 2,
            CacheFormat.Int8 => (dim + 3) / 4,
            _ => dim,
        };
        var part = lane.ShiftRight(dimShift);                                // this invocation's share of the positions
        var parts = k.Int(lanes).ShiftRight(dimShift);
        var d = lane & (k.Int(1).ShiftLeft(dimShift) - 1);

        // Dimension dd of cached row `slot` (= head · capacity + position) of a cache.
        Val At(Buf cache, Val slot, Val dd) => format switch
        {
            CacheFormat.BFloat16 => BFloat16At(k, cache, slot * words + (dd >> 1), dd),
            CacheFormat.Int8 => Int8At(k, cache, slot * words, dd),
            _ => cache[slot * dim + dd],
        };

        // q · k_slot · scale, a word of the key row at a time (four words per step, loaded before they are used). Past
        // dim, query[] holds zeros: the int8 row's padding bytes add +0; bfloat16's odd last half is never read.
        Val Score(Val slot)
        {
            var dot = k.Local(0f);
            var rowStart = slot * words;
            void Word(Val w, Val bits)
            {
                switch (format)
                {
                    case CacheFormat.Int8:
                        for (int b = 0; b < 4; b++)
                        {
                            dot.V = dot.V + query[4 * w + b] * ((bits << (24 - 8 * b)) >> 24).ToFloat();
                        }

                        break;
                    case CacheFormat.BFloat16:
                        dot.V = dot.V + query[2 * w] * (bits.AsUInt() << 16).AsFloat();
                        dot.V = dot.V + query[2 * w + 1] * (bits.AsUInt() & k.UInt(0xFFFF0000u)).AsFloat();
                        break;
                    default:
                        dot.V = dot.V + query[w] * bits.AsFloat();
                        break;
                }
            }

            // Words holding only dimensions below dim (bfloat16: the odd last dimension is added on its own).
            var whole = format == CacheFormat.BFloat16 ? dim >> 1 : words;
            Unrolled(k, whole, 4, w => keys.Int(rowStart + w), Word);
            if (format == CacheFormat.BFloat16)
            {
                k.If((dim & 1).Eq(1), () => dot.V = dot.V + query[dim - 1] * (keys.UInt(rowStart + whole) << 16).AsFloat());
            }

            return Capped(k, keyScales is null ? dot.V * scale : dot.V * keyScales[slot] * scale, softcap);
        }

        var acc = new Var[dimsPerLane];
        for (int j = 0; j < dimsPerLane; j++)
        {
            acc[j] = k.Local(ScalarKind.Float);
        }

        EachRow(k, heads * rowsPerHead, row =>
        {
            var h = row / rowsPerHead;
            var count = k.Min(start + row % rowsPerHead % steps, capacity - 1) + 1;
            var lo = WindowStart(k, count, window);
            var first = h * capacity;
            var chunk = k.Max((count - lo + splits - 1) / splits, k.Int(AttentionMinChunk(lanes)));
            var begin = lo + k.GroupY * chunk;
            var end = k.Min(count, begin + chunk);
            for (int j = 0; j < dimsPerLane; j++)
            {
                var at = lane + j * lanes;
                query[at] = k.Float(0f);
                k.If(at < dim, () => query[at] = q[row * dim + at]);
                acc[j].V = k.Float(0f);
            }

            k.Barrier();
            var max = k.Local(float.NegativeInfinity);                      // uniform: every invocation computes the same
            var total = k.Local(0f);
            k.For(begin, end, lanes, c0 =>
            {
                var c = c0 + lane;
                var score = k.Local(float.NegativeInfinity);
                k.If(c < end, () => score.V = Score(first + c));
                var tileMax = k.ReduceMax(scratch, score.V);
                var newMax = k.Max(max.V, tileMax);
                var rescale = k.Exp(max.V - newMax);                          // 0 for the first tile (max = -∞)
                var e = k.Local(0f);
                k.If(c < end, () => e.V = k.Exp(score.V - newMax));
                weights[lane] = valueScales is null ? e.V : e.V * valueScales[first + k.Min(c, end - 1)];
                total.V = k.Fma(total.V, rescale, e.V);
                max.V = newMax;
                k.Barrier();
                var tile = k.Min(end - c0, k.Int(lanes));
                var rowAt = first + c0 + part;
                var mine = (tile - part + parts - 1) / parts;                // this part's positions of the tile
                for (int j = 0; j < dimsPerLane; j++)
                {
                    var (sum, dj) = (k.Local(acc[j].V * rescale), d + j * lanes);
                    k.If(dj < dim, () => Unrolled(k, mine, 4, i => At(values, rowAt + i * parts, dj),
                        (i, v) => sum.V = k.Fma(weights[part + i * parts], v, sum.V)));
                    acc[j].V = sum.V;
                }

                k.Barrier();                                                  // weights read before the next tile writes them
            });

            var all = k.ReduceSum(scratch, total.V);
            for (int j = 0; j < dimsPerLane; j++)
            {
                if (j > 0)
                {
                    k.Barrier();                                              // the previous dimensions' sums are read
                }

                scratch[lane] = acc[j].V;
                k.Barrier();
                var dj = lane + j * lanes;
                k.If((lane < k.Int(1).ShiftLeft(dimShift)) & (dj < dim), () =>
                {
                    var value = k.Local(scratch[lane]);
                    k.For(k.Int(1), parts, 1, p => value.V = value.V + scratch[p.ShiftLeft(dimShift) + lane]);
                    k.If(splits.Eq(1), () => y[row * dim + dj] = value.V / all, () =>
                    {
                        var at = (row * splits + k.GroupY) * (dim + 2);
                        y[at + dj] = value.V;
                        k.If(dj.Eq(0), () =>
                        {
                            y[at + dim] = max.V;
                            y[at + dim + 1] = all;
                        });
                    });
                });
            }

            k.Barrier();                                                      // scratch and query are free for the next row
        });
        return k.Build();
    }

    // The first position a query whose causal limit is `end` (exclusive) sees: end - window, at least 0, with a window
    // (window > 0); else 0 (see AttentionVariant).
    private static Val WindowStart(KernelBuilder k, Val end, Val window) => k.Select(window > 0, k.Max(end - window, k.Int(0)), k.Int(0));

    // A scaled score soft-capped when cap > 0, cap · tanh(score / cap); else the score itself.
    private static Val Capped(KernelBuilder k, Val score, Val cap) => k.Select(cap > 0f, Tanh(k, score / cap) * cap, score);

    // The derivative of Capped given its result: 1 - (capped / cap)² when cap > 0, else 1.
    private static Val CapSlope(KernelBuilder k, Val capped, Val cap)
    {
        var ratio = capped / cap;
        return k.Select(cap > 0f, 1f - ratio * ratio, k.Float(1f));
    }

    // for (i = 0; i < count; i++) use(i, load(i)), `unroll` at a time with their loads first, then the rest one by one.

    private static void Unrolled(KernelBuilder k, Val count, int unroll, Func<Val, Val> load, Action<Val, Val> use)
    {
        var i = k.Local(k.Int(0));
        k.While(() => i.V + (unroll - 1) < count, () =>
        {
            var at = i.V;
            var loaded = new Val[unroll];
            for (int u = 0; u < unroll; u++)
            {
                loaded[u] = load(at + u);
            }

            for (int u = 0; u < unroll; u++)
            {
                use(at + u, loaded[u]);
            }

            i.V = at + unroll;
        });
        k.While(() => i.V < count, () =>
        {
            var at = i.V;
            use(at, load(at));
            i.V = at + 1;
        });
    }
}
