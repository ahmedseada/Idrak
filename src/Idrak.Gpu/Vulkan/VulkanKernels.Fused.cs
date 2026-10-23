// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Vulkan;

// Fused decoding kernels (Backend.FusedKernels): several packed products sharing one input in one dispatch (the query,
// key and value projections), the gate and up projections with the gated activation, a down projection reading the
// activation of its input, a projection's residual addition and the next RMS normalization, and the attention layer's
// norms, rotary positions and head layout (or cache writes) in one row pass. The products read the packed weights as
// PackedGemv does (same layouts, slices and the packed products' one order, VulkanKernels.Order.cs); products split over
// k write their partial sums for a second pass that finishes the same tree and the outputs, so every choice of width,
// words and splits gives the same bits.
internal static partial class VulkanKernels
{
    /// <summary>The fused packed-product kernels: several products sharing an input, the gate/up pair, and the gated down projection.</summary>
    internal enum FusedMode
    {
        Many,
        Pair,
        Gated,
    }

    /// <summary>Largest head size of <c>norm_rope_heads</c> (its row is staged in workgroup memory).</summary>
    public const int NormRopeMaxCols = AttentionMaxDim;

    /// <summary>The fused packed-product kernel for a mode, format, row block and words per row of k.</summary>
    public static string FusedGemvName(FusedMode mode, PackedFormat format, int rows, int words) => format switch
    {
        PackedFormat.Int8 => "int8_",
        PackedFormat.Int4 => "int4_",
        _ => "bf16_",
    } + mode switch
    {
        FusedMode.Many => "gemv_many_",
        FusedMode.Pair => "gemv_pair_",
        _ => "gemv_gated_",
    } + rows + (words == GemvWordCounts[0] ? "" : "_w" + words);

    private static IEnumerable<(string, Func<SpirvKernel>)> FusedKernels()
    {
        foreach (var mode in new[] { FusedMode.Many, FusedMode.Pair, FusedMode.Gated })
        {
            foreach (var format in new[] { PackedFormat.Int8, PackedFormat.Int4, PackedFormat.BFloat16 })
            {
                foreach (int rows in GemvRowBlocks)
                {
                    foreach (int words in GemvWordCounts)
                    {
                        string name = FusedGemvName(mode, format, rows, words);
                        yield return (name, () => FusedGemv(name, mode, format, rows, words));
                    }
                }
            }
        }

        // The second pass of split gemv_many / gemv_pair dispatches: part [splits, products, m, nmax] holds the partial
        // sums (product p's columns j < n_p of row r at ((s · products + p) · m + r) · nmax + j). Output (p, r, j) adds
        // its splits in their pairwise tree (aligned subtrees of k's leaves), times scales_p[j] when flags bit 3 (int8), plus bias_p[j] when flags bit p, into
        // y_p[r · n_p + j]. With flags bit 4 (the gate/up pair, products = 2, n_0 = n_1 = nmax) one invocation takes
        // both products' (r, j) and also writes hidden[r · nmax + j] = act(gate) · up (activation 0 SiLU, 1 GELU, else
        // ReLU). One invocation per output (pair: per column of a row).
        yield return ("gemv_many_reduce", () =>
        {
            var k = new KernelBuilder("gemv_many_reduce", Block);
            var part = k.Buffer("part");
            var (s0, s1, s2) = (k.Buffer("scales0"), k.Buffer("scales1"), k.Buffer("scales2"));
            var (b0, b1, b2) = (k.Buffer("bias0"), k.Buffer("bias1"), k.Buffer("bias2"));
            var (y0, y1, y2) = (k.Buffer("y0"), k.Buffer("y1"), k.Buffer("y2"));
            var hidden = k.Buffer("hidden");
            var (m, nmax, n0, n1, n2) = (k.PushInt("m"), k.PushInt("nmax"), k.PushInt("n0"), k.PushInt("n1"), k.PushInt("n2"));
            var (products, splits, flags, activation) = (k.PushInt("products"), k.PushInt("splits"), k.PushInt("flags"), k.PushInt("activation"));
            var scaled = (flags & 8).Ne(0);
            var pair = (flags & 16).Ne(0);
            var stack = new PairwiseStack(k, 1, SplitStackEntries, SumsMerge(k));

            // Product p's finished output (r, j): its splits in the pairwise tree (they are aligned subtrees of k's leaves).
            Val Finish(int p, Buf scales, Buf bias, Val r, Val j)
            {
                var total = Pairwise(k, stack, splits, s => part[((s * products + p) * m + r) * nmax + j]);
                var value = k.Local(k.Select(scaled, k.MulRounded(total, scales[j]), total));
                k.If((flags & (1 << p)).Ne(0), () => value.V = k.AddRounded(value.V, bias[j]));
                return value.V;
            }

            Grid(k, k.Select(pair, m * nmax, products * m * nmax), i =>
            {
                k.If(pair, () =>
                {
                    var (r, j) = (i / nmax, i % nmax);
                    var gate = Finish(0, s0, b0, r, j);
                    var up = Finish(1, s1, b1, r, j);
                    y0[r * nmax + j] = gate;
                    y1[r * nmax + j] = up;
                    var (value, _) = Activation(k, gate, activation, slope: false);
                    hidden[r * nmax + j] = value.V * up;
                }, () =>
                {
                    var p = i / (m * nmax);
                    var rest = i % (m * nmax);
                    var (r, j) = (rest / nmax, rest % nmax);
                    k.If(p.Eq(0), () => k.If(j < n0, () => y0[r * n0 + j] = Finish(0, s0, b0, r, j)),
                        () => k.If(p.Eq(1), () => k.If(j < n1, () => y1[r * n1 + j] = Finish(1, s1, b1, r, j)),
                            () => k.If(j < n2, () => y2[r * n2 + j] = Finish(2, s2, b2, r, j))));
                });
            });
            return k.Build();
        });

        // The second pass of PackedMatMulAddRmsNorm, per row r of [rows, cols]: v = Σ_s part[(s · rows + r) · cols + j]
        // over the splits in their pairwise tree (times scales[j] when scaled ≠ 0: int8 partial sums), written to y; sum = residual + v;
        // normalized = sum · inv · (gain[j] + offset), inv = 1 / sqrt(mean(sum²) + eps) (as add_rms_norm_affine). With one
        // split part is y itself (each column read and written by the same invocation).
        foreach (bool narrow in new[] { false, true })
        {
            string name = narrow ? "gemv_add_rms_norm_narrow" : "gemv_add_rms_norm";
            yield return (name, () =>
            {
                var k = new KernelBuilder(name, Block);
                var (part, scales, residual, y) = (k.Buffer("part"), k.Buffer("scales"), k.Buffer("residual"), k.Buffer("y"));
                var (sum, gain, normalized) = (k.Buffer("sum"), k.Buffer("gain"), k.Buffer("normalized"));
                var (rows, cols, splits, scaled) = (k.PushInt("rows"), k.PushInt("cols"), k.PushInt("splits"), k.PushInt("scaled"));
                var (eps, offset) = (k.PushFloat("eps"), k.PushFloat("offset"));
                var w = new RowWork(k, narrow);
                var stack = new PairwiseStack(k, 1, SplitStackEntries, SumsMerge(k));
                w.Each(rows, r =>
                {
                    var o = r * cols;
                    var squares = w.RowSumOfSquares(cols, j =>
                    {
                        var total = Pairwise(k, stack, splits, s => part[(s * rows + r) * cols + j]);
                        var v = k.Select(scaled.Ne(0), k.MulRounded(total, scales[j]), total);
                        y[o + j] = v;
                        var s = k.AddRounded(residual[o + j], v);
                        sum[o + j] = s;
                        return s;
                    });
                    var inverse = 1f / k.Sqrt(squares / cols.ToFloat() + eps);
                    w.Columns(cols, j => normalized[o + j] = sum[o + j] * inverse * (gain[j] + offset));
                });
                return k.Build();
            });
        }

        // The attention layer's projections in the layouts attention reads (Backend.NormRopeHeads): rows1 query rows of q,
        // then rows2 key rows of k and rows2 value rows of v (row = (b · steps + s) · heads + h, heads = kvHeads for keys
        // and values). A workgroup per row: the row is staged in workgroup memory, query and key rows RMS-normalized with
        // their gain when flags bit 0 (as rms_norm_rope: x · inv · (gain + offset)) and turned by the rotary tables (pairs
        // p < half, as rope; half 0: none); values copied. Queries go to yq [batch, heads, steps, cols]; keys and values to
        // yk / yv [batch, kvHeads, capacity, stride] at row offset + s (offset = position[0] when flags bit 2, else 0), as
        // floats or, when flags bit 1, as bfloat16 words (stride counts halves; rounded as key_value_write_bf16).
        //
        // Workgroup memory: row[j] is written by invocation j % width while loading (the reduction's barriers follow), then
        // each element by one invocation while normalizing and turning (pairs (i, j) by the invocation of p, the rest by
        // the invocation of their index), read for the stores after a barrier; a barrier ends the row.
        yield return ("norm_rope_heads", () =>
        {
            var k = new KernelBuilder("norm_rope_heads", Block);
            var (q, key, value) = (k.Buffer("q"), k.Buffer("k"), k.Buffer("v"));
            var (gainQ, gainK) = (k.Buffer("gainQ"), k.Buffer("gainK"));
            var (cos, sin, positions, position) = (k.Buffer("cos"), k.Buffer("sin"), k.Buffer("positions"), k.Buffer("position"));
            var (yq, yk, yv) = (k.Buffer("yq"), k.Buffer("yk"), k.Buffer("yv"));
            var (rows1, rows2, heads, kvHeads, steps) = (k.PushInt("rows1"), k.PushInt("rows2"), k.PushInt("heads"), k.PushInt("kvHeads"), k.PushInt("steps"));
            var (cols, half, interleaved, flags) = (k.PushInt("cols"), k.PushInt("half"), k.PushInt("interleaved"), k.PushInt("flags"));
            var (capacity, stride) = (k.PushInt("capacity"), k.PushInt("stride"));
            var (epsQ, offsetQ, epsK, offsetK) = (k.PushFloat("epsQ"), k.PushFloat("offsetQ"), k.PushFloat("epsK"), k.PushFloat("offsetK"));
            var row = k.Shared("row", NormRopeMaxCols);
            var scratch = k.Shared("scratch", Block);
            var lane = k.LocalX;
            var (normed, bfloat16, cached) = ((flags & 1).Ne(0), (flags & 2).Ne(0), (flags & 4).Ne(0));
            var pairs = interleaved.Ne(0);
            var start = k.Local(k.Int(0));
            k.If(cached, () => start.V = position[k.Int(0)].ToInt());
            EachRow(k, rows1 + 2 * rows2, at =>
            {
                var segment = k.Select(at < rows1, k.Int(0), k.Select(at < rows1 + rows2, k.Int(1), k.Int(2)));
                var r = at - k.Select(segment.Eq(0), k.Int(0), k.Select(segment.Eq(1), rows1, rows1 + rows2));
                var perStep = k.Select(segment.Eq(0), heads, kvHeads);
                var (h, t) = (r % perStep, r / perStep);
                var (s, b) = (t % steps, t / steps);
                var squares = k.Local(0f);
                k.For(lane, cols, Block, j =>
                {
                    var x = k.Local(ScalarKind.Float);
                    k.If(segment.Eq(0), () => x.V = q[r * cols + j], () => k.If(segment.Eq(1), () => x.V = key[r * cols + j], () => x.V = value[r * cols + j]));
                    row[j] = x.V;
                    squares.V = k.Fma(x.V, x.V, squares.V);
                });
                var total = k.ReduceSum(scratch, squares.V);                  // its barriers also order the row's stores
                var norm = normed & (segment < 2);
                var query = segment.Eq(0);
                var scale = k.Select(norm, 1f / k.Sqrt(total / cols.ToFloat() + k.Select(query, epsQ, epsK)), k.Float(1f));
                var shift = k.Select(query, offsetQ, offsetK);
                var turned = k.Select(segment < 2, half, k.Int(0));

                // (gain[j] + offset), or 1 without a norm.
                Val Gain(Val j)
                {
                    var g = k.Local(1f);
                    k.If(norm, () => k.If(query, () => g.V = gainQ[j] + shift, () => g.V = gainK[j] + shift));
                    return g.V;
                }

                k.For(lane, turned, Block, p =>
                {
                    var position = positions[s].ToInt();
                    var i = k.Select(pairs, 2 * p, p);
                    var j = k.Select(pairs, 2 * p + 1, p + half);
                    var a = row[i] * scale * Gain(i);
                    var c = row[j] * scale * Gain(j);
                    var co = cos[position * half + p];
                    var si = sin[position * half + p];
                    row[i] = k.Fma(a, co, -(c * si));
                    row[j] = k.Fma(c, co, a * si);
                });
                k.For(lane, cols - 2 * turned, Block, e =>
                {
                    var j = 2 * turned + e;
                    row[j] = row[j] * scale * Gain(j);
                });
                k.Barrier();
                k.If(query, () =>
                {
                    var first = ((b * heads + h) * steps + s) * cols;
                    k.For(lane, cols, Block, j => yq[first + j] = row[j]);
                }, () =>
                {
                    var slot = (b * kvHeads + h) * capacity + start.V + s;
                    var isKey = segment.Eq(1);
                    k.If(bfloat16, () =>
                    {
                        var first = slot * (stride >> 1);
                        k.For(lane, (cols + 1) / 2, Block, w =>
                        {
                            var low = RoundBFloat16(k, row[2 * w]);
                            var high = k.Local(k.UInt(0));
                            k.If(2 * w + 1 < cols, () => high.V = RoundBFloat16(k, row[2 * w + 1]));
                            var word = (low | high.V << 16).AsFloat();
                            k.If(isKey, () => yk[first + w] = word, () => yv[first + w] = word);
                        });
                    }, () =>
                    {
                        var first = slot * stride;
                        k.For(lane, cols, Block, j => k.If(isKey, () => yk[first + j] = row[j], () => yv[first + j] = row[j]));
                    });
                });
                k.Barrier();                                                  // the row is read before the next one is staged
            });
            return k.Build();
        });
    }

    // The fused packed products, with PackedGemv's grid, slices and sums (see there): x = column blocks of L words, y =
    // splits of k (chunk a multiple of 32), z = row blocks (Many: row blocks × products, product = z % products).
    //   Many: products 0 … products - 1 (weights q_p, scales_p, bias_p when flags bit p, output y_p of n_p columns); a
    //         workgroup takes one product, and nothing where its column block is past that product's columns.
    //   Pair: the gate and up products (n columns each) in turn in one workgroup, its sums of the gate kept until the
    //         up product's are added, then y0 = gate, y1 = up and hidden = act(gate) · up (activation: 0 SiLU, 1 GELU,
    //         else ReLU).
    //   Gated: one product of x = act(gate) · up, computed as each value is read.
    // One split writes the finished outputs (int8 times its column scale, plus the bias); several write the partial sums
    // to part (Many and Pair: [splits, products, m, nmax], for gemv_many_reduce; Gated: [splits, m, n] in y, for
    // gemv_reduce / int8_gemv_reduce).
    //
    // Workgroup memory: partial, as PackedGemv; a product's first round waits at a barrier for the previous product's
    // reads (Pair).
    private static SpirvKernel FusedGemv(string name, FusedMode mode, PackedFormat format, int rows, int lanes)
    {
        bool scaled = format != PackedFormat.BFloat16;
        var k = new KernelBuilder(name, Block);
        int products = mode switch { FusedMode.Many => 3, FusedMode.Pair => 2, _ => 1 };
        Buf x, up = null!;
        if (mode == FusedMode.Gated)
        {
            (x, up) = (k.Buffer("gate"), k.Buffer("up"));
        }
        else
        {
            x = k.Buffer("x");
        }

        var q = new Buf[products];
        var scales = new Buf?[products];
        var bias = new Buf[products];
        var y = new Buf[products];
        for (int p = 0; p < products; p++)
        {
            q[p] = k.Buffer(format == PackedFormat.BFloat16 ? "packed" + p : "q" + p);
        }

        for (int p = 0; p < products && scaled; p++)
        {
            scales[p] = k.Buffer("scales" + p);
        }

        if (mode != FusedMode.Gated)
        {
            for (int p = 0; p < products; p++)
            {
                bias[p] = k.Buffer("bias" + p);
            }
        }

        for (int p = 0; p < products; p++)
        {
            y[p] = k.Buffer("y" + p);
        }

        var hidden = mode == FusedMode.Pair ? k.Buffer("hidden") : null;
        var part = mode == FusedMode.Gated ? null : k.Buffer("part");
        Val m, depth, chunk, flags = default, activation = default, productCount = default;
        var n = new Val[products];
        if (mode == FusedMode.Many)
        {
            (m, depth, chunk) = (k.PushInt("m"), k.PushInt("k"), k.PushInt("chunk"));
            for (int p = 0; p < products; p++)
            {
                n[p] = k.PushInt("n" + p);
            }

            (productCount, flags) = (k.PushInt("products"), k.PushInt("flags"));
        }
        else
        {
            (m, n[0], depth, chunk) = (k.PushInt("m"), k.PushInt("n"), k.PushInt("k"), k.PushInt("chunk"));
            for (int p = 1; p < products; p++)
            {
                n[p] = n[0];
            }

            if (mode == FusedMode.Pair)
            {
                flags = k.PushInt("flags");
            }

            activation = k.PushInt("activation");
        }

        var nmax = mode == FusedMode.Many ? k.Max(n[0], k.Max(n[1], n[2])) : n[0];
        var groupZ = mode == FusedMode.Many ? k.GroupZ / productCount : k.GroupZ;
        var single = k.GroupsY.Eq(1);
        // Gated: the activation is computed as each input value is read, so the loads are not unrolled (each unrolled
        // load would carry its own copy of the activation's code).
        var body = new GemvBody(k, format, rows, lanes, m, depth, chunk, groupZ, unroll: mode != FusedMode.Gated, mode == FusedMode.Gated
            ? i =>
            {
                var (v, _) = Activation(k, x[i], activation, slope: false);
                return v.V * up[i];
            }
            : i => x[i]);

        // A product's finished value of column j: int8 times the column's scale, plus the bias when its flag is set.
        Val Finish(int p, Val total, Val j)
        {
            var value = k.Local(format == PackedFormat.Int8 ? k.MulRounded(total, scales[p]![j]) : total);
            if (mode != FusedMode.Gated)
            {
                k.If((flags & (1 << p)).Ne(0), () => value.V = k.AddRounded(value.V, bias[p][j]));
            }

            return value.V;
        }

        switch (mode)
        {
            case FusedMode.Many:
            {
                var p = k.GroupZ % productCount;
                void Product(int index) => k.If(k.GroupX * lanes < (n[index] + (ColumnsPerWord(format) - 1)) / ColumnsPerWord(format), () =>
                    body.Fresh().Product(q[index], scales[index], n[index], (_, row, j, total) => k.If(single,
                        () => y[index][row * n[index] + j] = Finish(index, total, j),
                        () => part![((k.GroupY * productCount + index) * m + row) * nmax + j] = total)));
                k.If(p.Eq(0), () => Product(0), () => k.If(p.Eq(1), () => Product(1), () => Product(2)));
                break;
            }

            case FusedMode.Pair:
            {
                var saved = new Var[body.Slots];
                for (int i = 0; i < saved.Length; i++)
                {
                    saved[i] = k.Local(0f);
                }

                body.Product(q[0], scales[0], n[0], (slot, _, _, total) => saved[slot].V = total);
                body.Product(q[1], scales[1], n[1], (slot, row, j, total) => k.If(single, () =>
                {
                    var gate = Finish(0, saved[slot].V, j);
                    var upValue = Finish(1, total, j);
                    var o = row * n[0] + j;
                    y[0][o] = gate;
                    y[1][o] = upValue;
                    var (value, _) = Activation(k, gate, activation, slope: false);
                    hidden![o] = value.V * upValue;
                }, () =>
                {
                    part![((k.GroupY * 2) * m + row) * n[0] + j] = saved[slot].V;
                    part![((k.GroupY * 2 + 1) * m + row) * n[0] + j] = total;
                }));
                break;
            }

            default:
                body.Product(q[0], scales[0], n[0], (_, row, j, total) =>
                    y[0][(k.GroupY * m + row) * n[0] + j] = format == PackedFormat.Int8 ? k.Select(single, k.MulRounded(total, scales[0]![j]), total) : total);
                break;
        }

        return k.Build();
    }

    /// <summary>
    /// Entries of a packed-product kernel's pairwise stack (GemvBody), the top (in registers) included: at most 64 floats
    /// (256 bytes) in function memory per invocation, so a slice walks at most 2^entries leaves (the backend's GemvPlan
    /// keeps it within that; twice the memory measured no faster on a CPU driver).
    /// </summary>
    public static int GemvStackEntries(PackedFormat format, int rows) => Math.Clamp(64 / (rows * ColumnsPerWord(format)) + 1, 1, 16);

    // One packed product's sums for a workgroup's column block, split and row block in the packed products' one order
    // (VulkanKernels.Order.cs), then given to `emit(slot, row, j, total)` for each output of the block within m and the
    // product's n (slot: which of the invocation's Slots outputs it is, the same for every product).
    //
    // The split's rows [split · chunk, min(k, (split + 1) · chunk)) are leaves of `leaf` rows (chunk a power of two of
    // them; leaf a multiple of 32); slice t of the workgroup's S = W / L slices takes leaves [t · per, (t + 1) · per),
    // per = max(1, chunk / (leaf · S)), walking them in turn: a leaf's terms by fused multiply-adds from +0, row by row
    // in groups of 32 (int4's scale groups; full groups `unroll` rows a step, their words loaded before any is used),
    // the leaves through a PairwiseStack. Then the slices' totals, `batch` sums per lane per round through workgroup
    // memory (two barriers a round), in the pairwise tree of consecutive slices. With `sequential` set (the prompt-sized
    // products' order: one leaf of all of k, one split) slice t instead takes its own block of rows of x, block
    // rowBlock · S + t, and each invocation gives its own sums.
    //
    // The weights' binding (and int4's scales') starts at row first · chunk of k (a window of weights larger than a
    // binding); otherwise first is 0. Lanes past the last word read the last word again and give nothing; rows of the
    // block past m read row m - 1 and give nothing.
    //
    // Workgroup memory: partial[b · W + slice · L + lane] is written by invocation (slice, lane) only, after a barrier
    // that ends the previous round's (or product's) reads; it is read after the next barrier.
    private sealed class GemvBody
    {
        private readonly KernelBuilder _k;
        private readonly PackedFormat _format;
        private readonly int _rows, _lanes, _threads, _slices, _laneShift, _perWord, _sums, _batch;
        private readonly Val _m, _baseRow, _leaf, _unitBegin, _unitEnd, _rowBase, _lane, _tid;
        private readonly Val? _sequential;
        private readonly Val[] _xRows;
        private readonly Var[] _acc;
        private readonly Func<Val, Val> _x;
        private readonly SharedArray _partial;
        private readonly PairwiseStack _stack;
        private readonly bool _unroll;
        private bool _used;

        public GemvBody(KernelBuilder k, PackedFormat format, int rows, int lanes, Val m, Val depth, Val chunk, Val rowBlock, bool unroll, Func<Val, Val> x,
            Val? firstSplit = null, Val? leaf = null, Val? sequential = null)
        {
            (_k, _format, _rows, _lanes, _m, _unroll, _x, _sequential) = (k, format, rows, lanes, m, unroll, x, sequential);
            _threads = Block;
            _slices = _threads / lanes;
            _laneShift = System.Numerics.BitOperations.Log2((uint)lanes);
            int sliceShift = System.Numerics.BitOperations.Log2((uint)_slices);
            _perWord = ColumnsPerWord(format);
            _sums = rows * _perWord;
            _batch = Math.Min(_sums, 8);
            _partial = k.Shared("partial", _threads * _batch);
            _tid = k.LocalX;
            _lane = _tid & (lanes - 1);
            var slice = _tid >> _laneShift;
            var first = firstSplit ?? k.Int(0);
            _baseRow = first * chunk;                                          // the row of k the weights' binding starts at
            var kBegin = (first + k.GroupY) * chunk;
            var kEnd = k.Min(depth, kBegin + chunk);
            _leaf = leaf ?? k.Int(GemvChunkAlign);
            var per = k.Max(k.Int(1), (chunk / _leaf) >> sliceShift);
            var unitBegin = kBegin + slice * per * _leaf;
            var unitEnd = k.Min(kEnd, unitBegin + per * _leaf);
            var rowBase = rowBlock * rows;
            if (sequential is { } seq)
            {
                (unitBegin, unitEnd) = (k.Select(seq, kBegin, unitBegin), k.Select(seq, kEnd, unitEnd));
                rowBase = k.Select(seq, (rowBlock * _slices + slice) * rows, rowBase);
            }

            (_unitBegin, _unitEnd, _rowBase) = (unitBegin, unitEnd, rowBase);
            _xRows = new Val[rows];
            for (int r = 0; r < rows; r++)
            {
                _xRows[r] = k.Min(rowBase + r, m - 1) * depth;
            }

            _acc = new Var[_sums];
            for (int i = 0; i < _sums; i++)
            {
                _acc[i] = k.Local(0f);
            }

            _stack = new PairwiseStack(k, _sums, GemvStackEntries(format, rows), SumsMerge(k));
        }

        // Outputs per invocation and product: rounds × passes of the slices' sums.
        private int Passes => (_lanes * _batch + _threads - 1) / _threads;

        public int Slots => _sums / _batch * Passes;

        // The next product is the first to use the workgroup memory at run time (one of exclusive branches).
        public GemvBody Fresh()
        {
            _used = false;
            return this;
        }

        public void Product(Buf q, Buf? scales, Val n, Action<int, Val, Val, Val> emit)
        {
            var k = _k;
            int perWord = _perWord;
            var words = (n + (perWord - 1)) / perWord;
            var wordAt = k.Min(k.GroupX * _lanes + _lane, words - 1);

            void Accumulate(Val kk, Val bits, Val[]? groupScales)
            {
                var w = new Val[perWord];
                for (int c = 0; c < perWord; c++)
                {
                    w[c] = _format switch
                    {
                        PackedFormat.Int8 => ((bits << (24 - 8 * c)) >> 24).ToFloat(),
                        PackedFormat.Int4 => ((bits << (28 - 4 * c)) >> 28).ToFloat() * groupScales![c],
                        _ => c == 0 ? (bits.AsUInt() << 16).AsFloat() : (bits.AsUInt() & k.UInt(0xFFFF0000u)).AsFloat(),
                    };
                }

                for (int r = 0; r < _rows; r++)
                {
                    var xv = _x(_xRows[r] + kk);
                    for (int c = 0; c < perWord; c++)
                    {
                        _acc[r * perWord + c].V = k.Fma(xv, w[c], _acc[r * perWord + c].V);
                    }
                }
            }

            Val Word(Val kk) => q.Int((kk - _baseRow) * words + wordAt);

            // The slice's leaves in turn, each from +0, into the pairwise stack.
            int unrollRows = _format == PackedFormat.Int4 ? 8 : 4;
            _stack.Reset();
            var count = k.Max(k.Int(0), (_unitEnd - _unitBegin + _leaf - 1) / _leaf);
            k.For(k.Int(0), count, 1, j =>
            {
                var leafBegin = _unitBegin + j * _leaf;
                var leafEnd = k.Min(leafBegin + _leaf, _unitEnd);
                foreach (var a in _acc)
                {
                    a.V = k.Float(0f);
                }

                k.For(leafBegin / 32, (leafEnd + 31) / 32, 1, g =>
                {
                    Val[]? groupScales = null;
                    if (_format == PackedFormat.Int4)
                    {
                        groupScales = new Val[perWord];
                        for (int c = 0; c < perWord; c++)
                        {
                            groupScales[c] = scales![(g - _baseRow / 32) * (words * 8) + wordAt * 8 + c];
                        }
                    }

                    var start = g * 32;
                    var end = k.Min(start + 32, leafEnd);
                    if (!_unroll)
                    {
                        k.For(start, end, 1, kk => Accumulate(kk, Word(kk), groupScales));
                        return;
                    }

                    k.If((start + 32).Eq(end), () => k.For(start, end, unrollRows, at =>
                    {
                        var loaded = new Val[unrollRows];
                        for (int u = 0; u < unrollRows; u++)
                        {
                            loaded[u] = Word(at + u);
                        }

                        for (int u = 0; u < unrollRows; u++)
                        {
                            Accumulate(at + u, loaded[u], groupScales);
                        }
                    }), () => k.For(start, end, 1, kk => Accumulate(kk, Word(kk), groupScales)));
                });
                _stack.Push(j, [.. _acc.Select(a => a.V)]);
            });
            var totals = _stack.Total([.. Enumerable.Repeat(k.Float(-0f), _sums)]);

            if (_sequential is { } seq)
            {
                k.If(seq, () =>
                {
                    for (int i = 0; i < _sums; i++)
                    {
                        int slot = i;
                        var (row, j) = (_rowBase + i / perWord, (k.GroupX * _lanes + _lane) * perWord + i % perWord);
                        var total = totals[i];
                        k.If((row < _m) & (j < n), () => emit(slot, row, j, total));
                    }
                }, () => Combine(totals, n, emit));
            }
            else
            {
                Combine(totals, n, emit);
            }
        }

        // The slices' totals in the pairwise tree of consecutive slices, `batch` per lane per round: output o < L · batch
        // (invocations o, o + W, …) combines sum b = o / L of lane o % L over the S slices.
        private void Combine(Val[] totals, Val n, Action<int, Val, Val, Val> emit)
        {
            var k = _k;
            int threads = _threads, lanes = _lanes, batch = _batch;
            for (int round = 0; round < _sums / batch; round++)
            {
                if (_used)
                {
                    k.Barrier();                                               // the previous round's (or product's) reads are done
                }

                _used = true;
                for (int b = 0; b < batch; b++)
                {
                    _partial[k.Int(b * threads) + _tid] = totals[round * batch + b];
                }

                k.Barrier();
                int first = round * batch;
                for (int pass = 0; pass * threads < lanes * batch; pass++)
                {
                    int slot = round * Passes + pass;
                    var o = _tid + pass * threads;
                    k.If(o < lanes * batch, () =>
                    {
                        var (l, b) = (o & (lanes - 1), o >> _laneShift);
                        Val Tree(int from, int count) => count == 1
                            ? _partial[b * threads + k.Int(from * lanes) + l]
                            : k.AddRounded(Tree(from, count / 2), Tree(from + count / 2, count / 2));
                        var total = Tree(0, _slices);
                        var index = first + b;
                        var (r, c) = (index / _perWord, index % _perWord);
                        var (row, j) = (_rowBase + r, (k.GroupX * lanes + l) * _perWord + c);
                        k.If((row < _m) & (j < n), () => emit(slot, row, j, total));
                    });
                }
            }
        }
    }
}
