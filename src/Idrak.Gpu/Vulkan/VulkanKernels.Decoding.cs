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

        // gather of columns from a bfloat16 table [dim, vocabulary] (rows of ⌈vocabulary / 2⌉ words): y[t, d] =
        // table[d, indices[t]], a tied head's weight read as the embedding table. The binding holds the table's rows
        // [first, first + rows) (rows of the table are the outputs' columns d; windows as in gather).
        yield return ("gather_bf16_columns", () =>
        {
            var k = new KernelBuilder("gather_bf16_columns", Block);
            var (packed, indices, y) = (k.Buffer("packed"), k.Buffer("indices"), k.Buffer("y"));
            var (count, dim, vocabulary) = (k.PushInt("count"), k.PushInt("dim"), k.PushInt("vocabulary"));
            var (first, rows) = (k.PushInt("first"), k.PushInt("rows"));
            Grid(k, count * dim, i =>
            {
                var (token, d) = (i / dim, i % dim - first);
                var index = k.Clamp(indices[token].ToInt(), k.Int(0), vocabulary - 1);
                k.If((d >= 0) & (d < rows), () => y[i] = BFloat16At(k, packed, d * ((vocabulary + 1) / 2) + (index >> 1), index));
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

        // attention_decode and attention_bf16 for rows of different lengths decoded together (Backend.AttentionRows,
        // AttentionRowsBFloat16): each row from its own start, starts[(h / headsPerRow) · steps + t] (floats); a row that
        // sees nothing (padding) gets zeros.
        yield return ("attention_rows_f32", () => Attention("attention_rows_f32", CacheFormat.Float, rows: true));
        yield return ("attention_rows_bf16", () => Attention("attention_rows_bf16", CacheFormat.BFloat16, rows: true));

        // Merges the splits of a split attention dispatch: part [rows, splits, dim + 2] holds each split's state (its
        // weighted values, then its max and total; total 0: no positions). y[row, d] = weighted values / total of the
        // splits merged in decoding attention's one order (AttentionMerge): the splits taken in bit-reversed order
        // through a PairwiseStack, which pairs them as the tree of the blocks' indices does; zeros for a row no split saw
        // (total 0: padding of attention_rows_*). One invocation per output.
        yield return ("attention_combine", () =>
        {
            var k = new KernelBuilder("attention_combine", Block);
            var (part, y) = (k.Buffer("part"), k.Buffer("y"));
            var (rows, splits, dim) = (k.PushInt("rows"), k.PushInt("splits"), k.PushInt("dim"));
            var stack = new PairwiseStack(k, 3, SplitStackEntries, (l, r) => AttentionMerge(k, l, r));
            var bits = k.FindMsb(k.Max(splits, k.Int(1)));
            Grid(k, rows * dim, i =>
            {
                var (row, d) = (i / dim, i % dim);
                var stride = dim + 2;
                stack.Reset();
                k.For(k.Int(0), splits, 1, j =>
                {
                    var s = BitReversed(k, j, bits);
                    var at = (row * splits + s) * stride;
                    stack.Push(j, [part[at + dim], part[at + dim + 1], part[at + d]]);
                });
                var state = stack.Total([k.Float(0f), k.Float(0f), k.Float(0f)]);
                y[i] = k.Select(state[1].Eq(k.Float(0f)), k.Float(0f), state[2] / state[1]);
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

    // `value`'s low `bits` bits in reverse order (0 for no bits).
    private static Val BitReversed(KernelBuilder k, Val value, Val bits) =>
        k.Select(bits.Eq(0), k.Int(0), k.BitReverse(value).AsUInt().ShiftRight((32 - bits).AsUInt()).AsInt());

    // Two attention states (max m, total t, weighted values a…) merged, the left one first: each rescaled to the larger
    // max, exp(m - max) · t and exp(m - max) · a added with the right one's product rounded on its own. A state with total
    // 0 (no positions) gives the other exactly; the max of an empty state is never read.
    private static Val[] AttentionMerge(KernelBuilder k, Val[] left, Val[] right)
    {
        var (m1, t1, m2, t2) = (left[0], left[1], right[0], right[1]);
        var m = k.Max(m1, m2);
        var (alpha1, alpha2) = (k.Exp(m1 - m), k.Exp(m2 - m));
        var (leftEmpty, rightEmpty) = (!(t1 > 0f), !(t2 > 0f));
        var result = new Val[left.Length];
        for (int i = 0; i < left.Length; i++)
        {
            var merged = i == 0 ? m : k.Fma(left[i], alpha1, k.MulRounded(right[i], alpha2));
            result[i] = k.Select(rightEmpty, left[i], k.Select(leftEmpty, right[i], merged));
        }

        return result;
    }

    // Attention over a key/value cache [heads, capacity, dim] for query rows q [heads, rowsPerHead, dim]: row i of head h
    // sees positions c = lo … count - 1, count = min(position[0] + i % rowsPerHead % steps, capacity - 1) + 1, lo =
    // max(count - window, 0) with a window (window > 0), else 0. Grid: x = query rows (a group takes rows x, x + groups,
    // …), y = splits (a power of two). With `rows` (attention_rows_*: rows of different lengths), lo is also at least the
    // row's own start, starts[(h / headsPerRow) · steps + t] for t = i % rowsPerHead % steps (a binding before the
    // position, a push constant after softcap), and a row that sees nothing (lo ≥ count: padding) gets zeros.
    //
    // In decoding attention's one order (VulkanKernels.Order.cs), whatever the width or the splits: the row's positions
    // in blocks of B = the head size rounded up to a power of two (block b: positions lo + b·B …, the last one shorter),
    // and per block its scores' max m (q · k_c · scale, each dot product a fused multiply-add per dimension in turn,
    // soft-capped when softcap > 0), the weights e_c = exp(score_c - m) (int8: times the value row's scale for the
    // values), their total t and the weighted values a[d] = Σ e_c · v_c[d], each in position order from +0. The blocks
    // merge (AttentionMerge) in the tree of their indices' bits from the lowest: split s takes the blocks b ≡ s (mod
    // splits); within it the workgroup's parts (P = W / B parts of B invocations, or one part of all W when B > W) take
    // the next bits, part p the blocks b = s + splits · (p + P · i); a part walks its i in bit-reversed order through a
    // PairwiseStack (one block per part a tile, the parts in step), then the parts merge in the same tree (part p with p
    // + P/2, …, through workgroup memory). Invocation (p, d) of a part scores the block's position d (and d + W, … when
    // B > W) and keeps its dimensions d (d + W, …); every invocation of a part keeps the part's max and total. The row
    // is written y = a / t with one split, else (a, m, t) to part[row, split] for attention_combine (t 0: no positions).
    //
    // Workgroup memory — every phase is separated by a barrier:
    //   query[max(256, W)] written at the start of a row (after the previous row's final barrier), read while scoring;
    //   scores[max(256, W)], weights (and int8's exps): written by the invocation of a position, read after a barrier;
    //   scratch[W]: the tile's maxima (shuffles, then a tree within each part, a barrier a level), then the parts' merge
    //   with scores and weights (each level writes, then after a barrier reads, then a barrier);
    //   totals[W]: a block's total by its part's first invocation, read by the part after the tile's last barrier and
    //   written again only after the next tile's barriers.
    private static SpirvKernel Attention(string name, CacheFormat format, bool rows = false)
    {
        int lanes = Block, dimsPerLane = Math.Max(1, AttentionMaxDim / lanes), widthBits = System.Numerics.BitOperations.Log2((uint)lanes);
        int staged = dimsPerLane * lanes;
        var k = new KernelBuilder(name, lanes);
        var (q, keys, values) = (k.Buffer("q"), k.Buffer("keys"), k.Buffer("values"));
        var (keyScales, valueScales) = format == CacheFormat.Int8 ? (k.Buffer("keyScales"), k.Buffer("valueScales")) : (null, null);
        var starts = rows ? k.Buffer("starts") : null;
        var (position, y) = (k.Buffer("position"), k.Buffer("y"));
        var (heads, rowsPerHead, steps, capacity, dim) = (k.PushInt("heads"), k.PushInt("rowsPerHead"), k.PushInt("steps"), k.PushInt("capacity"), k.PushInt("dim"));
        var (scale, _) = (k.PushFloat("scale"), k.PushInt("dimShift"));
        var (window, softcap) = (k.PushInt("window"), k.PushFloat("softcap"));
        Val? headsPerRow = rows ? k.PushInt("headsPerRow") : null;
        var query = k.Shared("query", staged);
        var scores = k.Shared("scores", staged);
        var weights = k.Shared("weights", staged);
        var exps = format == CacheFormat.Int8 ? k.Shared("exps", staged) : weights;
        var scratch = k.Shared("scratch", lanes);
        var totals = k.Shared("totals", lanes);
        var lane = k.LocalX;
        var start = position[k.Int(0)].ToInt();
        var splits = k.GroupsY;
        var splitBits = k.FindMsb(k.Max(splits, k.Int(1)));
        var words = format switch
        {
            CacheFormat.BFloat16 => (dim + 1) / 2,
            CacheFormat.Int8 => (dim + 3) / 4,
            _ => dim,
        };

        // Blocks of B = 2^blockBits positions; parts of 2^laneBits invocations.
        var blockBits = k.Select(dim > 1, k.FindMsb(dim - 1) + 1, k.Int(0));
        var laneBits = k.Min(blockBits, k.Int(widthBits));
        var block = k.Int(1).ShiftLeft(blockBits);
        var partLanes = k.Int(1).ShiftLeft(laneBits);
        var partBits = widthBits - laneBits;
        var parts = k.Int(1).ShiftLeft(partBits);
        var part = lane.ShiftRight(laneBits);
        var d = lane & (partLanes - 1);
        var perLane = block.ShiftRight(laneBits);                            // positions an invocation scores per block

        // Dimension dd of cached row `slot` (= head · capacity + position) of a cache.
        Val At(Buf cache, Val slot, Val dd) => format switch
        {
            CacheFormat.BFloat16 => BFloat16At(k, cache, slot * words + (dd >> 1), dd),
            CacheFormat.Int8 => Int8At(k, cache, slot * words, dd),
            _ => cache[slot * dim + dd],
        };

        // q · k_slot · scale, a word of the key row at a time (four words per step, loaded before they are used), each
        // term a fused multiply-add in turn. Past dim, query[] holds zeros: the int8 row's padding bytes add +0;
        // bfloat16's odd last half is never read.
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
                            dot.V = k.Fma(query[4 * w + b], ((bits << (24 - 8 * b)) >> 24).ToFloat(), dot.V);
                        }

                        break;
                    case CacheFormat.BFloat16:
                        dot.V = k.Fma(query[2 * w], (bits.AsUInt() << 16).AsFloat(), dot.V);
                        dot.V = k.Fma(query[2 * w + 1], (bits.AsUInt() & k.UInt(0xFFFF0000u)).AsFloat(), dot.V);
                        break;
                    default:
                        dot.V = k.Fma(query[w], bits.AsFloat(), dot.V);
                        break;
                }
            }

            // Words holding only dimensions below dim (bfloat16: the odd last dimension is added on its own).
            var whole = format == CacheFormat.BFloat16 ? dim >> 1 : words;
            Unrolled(k, whole, 4, w => keys.Int(rowStart + w), Word);
            if (format == CacheFormat.BFloat16)
            {
                k.If((dim & 1).Eq(1), () => dot.V = k.Fma(query[dim - 1], (keys.UInt(rowStart + whole) << 16).AsFloat(), dot.V));
            }

            return Capped(k, keyScales is null ? dot.V * scale : dot.V * keyScales[slot] * scale, softcap);
        }

        var acc = Enumerable.Range(0, dimsPerLane).Select(_ => k.Local(0f)).ToArray();
        var (blockMax, blockTotal) = (k.Local(0f), k.Local(0f));
        var stack = new PairwiseStack(k, 2 + dimsPerLane, 17, (l, r) => AttentionMerge(k, l, r));   // a part walks at most 2^16 blocks

        EachRow(k, heads * rowsPerHead, row =>
        {
            var h = row / rowsPerHead;
            var t = row % rowsPerHead % steps;
            var count = k.Min(start + t, capacity - 1) + 1;
            var lo = WindowStart(k, count, window);
            if (starts is not null)
            {
                lo = k.Min(k.Max(lo, starts[h / headsPerRow!.Value * steps + t].ToInt()), count);   // at count: no blocks (padding)
            }

            var first = h * capacity;
            var blocks = (count - lo + block - 1).ShiftRight(blockBits);
            var treeBits = k.Select(blocks > 1, k.FindMsb(blocks - 1) + 1, k.Int(0));
            var walkBits = k.Max(k.Int(0), treeBits - splitBits - partBits);  // a part's blocks: 2^walkBits in bit-reversed order
            for (int j = 0; j < dimsPerLane; j++)
            {
                var at = lane + j * lanes;
                query[at] = k.Float(0f);
                k.If(at < dim, () => query[at] = q[row * dim + at]);
            }

            stack.Reset();
            k.Barrier();
            k.For(k.Int(0), k.Int(1).ShiftLeft(walkBits), 1, j =>
            {
                var b = k.GroupY + (part + BitReversed(k, j, walkBits).ShiftLeft(partBits)).ShiftLeft(splitBits);
                var begin = lo + b.ShiftLeft(blockBits);
                var end = k.Min(count, begin + block);
                var used = k.Max(k.Int(0), end - begin);                     // the block's positions (0: past the row's)

                // 1. Scores of the block's positions, this invocation's largest.
                var mine = k.Local(float.MinValue);
                for (int u = 0; u < dimsPerLane; u++)
                {
                    var offset = d + u * lanes;
                    k.If((k.Int(u) < perLane) & (offset < used), () =>
                    {
                        var score = Score(first + begin + offset);
                        scores[part.ShiftLeft(blockBits) + offset] = score;
                        mine.V = k.Max(mine.V, score);
                    });
                }

                // 2. The block's max (any order gives its bits): xor shuffles within the part's subgroups, then the rest of
                // the part through workgroup memory.
                int butterfly = Math.Min(k.Butterfly, lanes);
                for (int stride = 1; stride < butterfly; stride *= 2)
                {
                    uint s = (uint)stride;
                    k.If(k.Int(stride) < partLanes, () => mine.V = k.Max(mine.V, k.SubgroupShuffleXor(mine.V, s)));
                }

                scratch[lane] = mine.V;
                k.Barrier();
                for (int stride = Math.Max(butterfly, 1); stride < lanes; stride *= 2)
                {
                    int s = stride;
                    k.If(k.Int(s) < partLanes, () =>
                    {
                        k.If((d & (2 * s - 1)).Eq(0), () => scratch[lane] = k.Max(scratch[lane], scratch[lane + s]));
                        k.Barrier();
                    });
                }

                blockMax.V = scratch[part.ShiftLeft(laneBits)];

                // 3. The weights.
                for (int u = 0; u < dimsPerLane; u++)
                {
                    var offset = d + u * lanes;
                    k.If((k.Int(u) < perLane) & (offset < used), () =>
                    {
                        var at = part.ShiftLeft(blockBits) + offset;
                        var e = k.Exp(scores[at] - blockMax.V);
                        exps[at] = e;
                        if (valueScales is not null)
                        {
                            weights[at] = k.MulRounded(e, valueScales[first + begin + offset]);
                        }
                    });
                }

                k.Barrier();

                // 4. The block's weighted values, position by position from +0, and its total (by the part's first
                // invocation along its first dimension, given to the part through workgroup memory).
                var row0 = part.ShiftLeft(blockBits);
                var leader = d.Eq(0);
                for (int jj = 0; jj < dimsPerLane; jj++)
                {
                    var (sum, dj) = (k.Local(0f), d + jj * lanes);
                    if (jj == 0)
                    {
                        var tally = k.Local(0f);
                        k.If(leader, () => Unrolled(k, used, 4, w => At(values, first + begin + w, dj), (w, v) =>
                        {
                            sum.V = k.Fma(weights[row0 + w], v, sum.V);
                            tally.V = k.AddRounded(tally.V, exps[row0 + w]);
                        }), () => k.If(dj < dim, () => Unrolled(k, used, 4, w => At(values, first + begin + w, dj),
                            (w, v) => sum.V = k.Fma(weights[row0 + w], v, sum.V))));
                        k.If(leader, () => totals[part] = tally.V);
                    }
                    else
                    {
                        k.If(dj < dim, () => Unrolled(k, used, 4, w => At(values, first + begin + w, dj),
                            (w, v) => sum.V = k.Fma(weights[row0 + w], v, sum.V)));
                    }

                    acc[jj].V = sum.V;
                }

                k.Barrier();                                                  // scores, weights and totals read after this
                blockTotal.V = totals[part];
                stack.Push(j, [blockMax.V, blockTotal.V, .. acc.Select(a => a.V)]);
            });

            var state = stack.Total([k.Float(0f), k.Float(0f), .. Enumerable.Repeat(k.Float(0f), dimsPerLane)]);

            // 5. The parts merged: part p with p + P/2, …, each level read, then written, through workgroup memory (maxima
            // in scratch, totals in weights, weighted values in scores).
            var merged = state.Select(v => k.Local(v)).ToArray();
            for (int stride = lanes / 2; stride > 0; stride /= 2)
            {
                int s = stride;
                var active = k.Int(s) < parts;
                k.If(active, () =>
                {
                    k.If(d.Eq(0), () =>
                    {
                        scratch[part] = merged[0].V;
                        weights[part] = merged[1].V;
                    });
                    for (int jj = 0; jj < dimsPerLane; jj++)
                    {
                        scores[lane + jj * lanes] = merged[2 + jj].V;
                    }
                });
                k.Barrier();
                var other = Enumerable.Range(0, 2 + dimsPerLane).Select(_ => k.Local(0f)).ToArray();
                var taking = active & (part < s);
                k.If(taking, () =>
                {
                    var partner = part + s;
                    other[0].V = scratch[partner];
                    other[1].V = weights[partner];
                    for (int jj = 0; jj < dimsPerLane; jj++)
                    {
                        other[2 + jj].V = scores[partner.ShiftLeft(laneBits) + d + jj * lanes];
                    }

                    var result = AttentionMerge(k, [.. merged.Select(v => v.V)], [.. other.Select(v => v.V)]);
                    for (int c = 0; c < merged.Length; c++)
                    {
                        merged[c].V = result[c];
                    }
                });
                k.Barrier();
            }

            // 6. The row (part 0 holds it).
            var (max, total) = (merged[0].V, merged[1].V);
            for (int jj = 0; jj < dimsPerLane; jj++)
            {
                var dj = d + jj * lanes;
                var value = merged[2 + jj].V;
                Val Single() => rows ? k.Select(total.Eq(k.Float(0f)), k.Float(0f), value / total) : value / total;   // padding: zeros
                k.If(part.Eq(0) & (dj < dim), () => k.If(splits.Eq(1), () => y[row * dim + dj] = Single(), () =>
                {
                    var at = (row * splits + k.GroupY) * (dim + 2);
                    y[at + dj] = value;
                    k.If(dj.Eq(0), () =>
                    {
                        y[at + dim] = max;
                        y[at + dim + 1] = total;
                    });
                }));
            }

            k.Barrier();                                                      // workgroup memory is free for the next row
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
