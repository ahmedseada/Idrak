// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Vulkan;

// Attention over one range of keys per query row (Backend.AttentionSpans), tiled as attention_tiled: the same
// workgroups, staging and online softmax, with each row's range read from the starts and ends buffers instead of
// computed from a causal limit and a window. Its gradient (Backend.AttentionSpansBackward) tiled the same way, the
// weights recomputed from each row's log-sum-exp: Δ = dOutput · output per row (attention_spans_delta), then a workgroup
// per block of query rows adds dq (attention_spans_backward_dq) and one per block of keys adds dkeys and dvalues through
// every query head of its key/value head's group (attention_spans_backward_dkv). Every gradient element has one owner
// that adds its terms in a fixed order: no atomics, the same sums every run.
internal static partial class VulkanKernels
{
    private static IEnumerable<(string, Func<SpirvKernel>)> SpanKernels()
    {
        yield return ("attention_spans", () => SpanAttention("attention_spans", logSumExp: false));
        yield return ("attention_spans_lse", () => SpanAttention("attention_spans_lse", logSumExp: true));
        yield return ("attention_spans_delta", SpanDelta);
        yield return ("attention_spans_backward_dq", SpanBackwardQueries);
        yield return ("attention_spans_backward_dkv", SpanBackwardKeys);
    }

    // y[h, i] = Σ_c softmax(cap(scale · q[h, i] · keys[g, c])) · values[g, c] over starts[t·rows + i] ≤ c < ends[t·rows + i]
    // (clamped to [0, keyRows]), g = h / group, t = h / headsPerTable; q, y [heads, rows, dim], keys, values [kvHeads,
    // keyRows, dim]. A workgroup of W invocations takes R = TiledAttentionRows(W) consecutive rows of one head (a block)
    // and goes through the keys C = W / R at a time, from the smallest start of its rows to their largest end, exactly as
    // TiledAttention does (see there for the phases and the workgroup memory); a key outside a row's range scores -∞ for
    // that row. A row whose range is empty keeps a max of -∞ and a total of 0: it gets zeros (and a log-sum-exp of -∞).
    private static SpirvKernel SpanAttention(string name, bool logSumExp)
    {
        int width = Block, rows = TiledAttentionRows(width), tile = TiledAttentionPositions(width);
        const int Depth = TiledAttentionDepth, Stride = Depth + 1;
        int outputs = AttentionMaxDim / tile, perStage = Depth / tile, stages = AttentionMaxDim / Depth;
        var k = new KernelBuilder(name, width);
        var (q, keys, values, starts, ends, y) = (k.Buffer("q"), k.Buffer("keys"), k.Buffer("values"), k.Buffer("starts"), k.Buffer("ends"), k.Buffer("y"));
        var lse = logSumExp ? k.Buffer("logSumExp") : null;
        var (heads, rowCount, keyRows, dim) = (k.PushInt("heads"), k.PushInt("rows"), k.PushInt("keyRows"), k.PushInt("dim"));
        var (group, headsPerTable) = (k.PushInt("group"), k.PushInt("headsPerTable"));
        var (scale, softcap) = (k.PushFloat("scale"), k.PushFloat("softcap"));
        var queries = k.Shared("queries", rows * Stride);
        var staged = k.Shared("staged", tile * Stride);
        var scores = k.Shared("scores", width);
        var maxes = k.Shared("max", rows);
        var alphas = k.Shared("alpha", rows);
        var totals = k.Shared("total", rows);
        var tid = k.LocalX;
        var (r, c) = (tid / tile, tid % tile);
        var blocksPerHead = (rowCount + (rows - 1)) / rows;
        var acc = new Var[outputs];
        for (int j = 0; j < outputs; j++)
        {
            acc[j] = k.Local(ScalarKind.Float);
        }

        k.For(k.GroupX, heads * blocksPerHead, block =>
        {
            var h = block / blocksPerHead;
            var r0 = block % blocksPerHead * rows;
            var first = h / group * keyRows;                                  // the key/value head's first row
            var table = h / headsPerTable * rowCount;                         // the head's table of ranges
            var mine = k.Min(r0 + r, rowCount - 1);                            // this invocation's row within the head
            var lo = k.Clamp(starts[table + mine].ToInt(), k.Int(0), keyRows);
            var hi = k.Clamp(ends[table + mine].ToInt(), k.Int(0), keyRows);

            // The keys the block sees: from the smallest start of its rows that see any to their largest end.
            var from = k.Local(keyRows);
            var count = k.Local(0);
            k.For(r0, k.Min(r0 + rows, rowCount), 1, row =>
            {
                var s = k.Clamp(starts[table + row].ToInt(), k.Int(0), keyRows);
                var e = k.Clamp(ends[table + row].ToInt(), k.Int(0), keyRows);
                k.If(e > s, () =>
                {
                    from.V = k.Min(from.V, s);
                    count.V = k.Max(count.V, e);
                });
            });

            var queryBase = h * rowCount;
            foreach (var v in acc)
            {
                v.V = k.Float(0f);
            }

            k.If(tid < rows, () =>
            {
                maxes[tid] = k.Float(float.NegativeInfinity);
                totals[tid] = k.Float(0f);
            });

            var end = count.V;
            k.For(from.V, end, tile, c0 =>
            {
                // 1. Scores, a stage of 64 dimensions at a time.
                var score = k.Local(0f);
                k.For(k.Int(0), dim, Depth, d0 =>
                {
                    k.For(tid, k.Int(rows * Depth), width, i =>
                    {
                        var (qr, e) = (i / Depth, i % Depth);
                        var value = k.Local(0f);
                        k.If(d0 + e < dim, () => value.V = q[(queryBase + k.Min(r0 + qr, rowCount - 1)) * dim + d0 + e]);
                        queries[qr * Stride + e] = value.V;
                    });
                    k.For(tid, k.Int(tile * Depth), width, i =>
                    {
                        var (kc, e) = (i / Depth, i % Depth);
                        var value = k.Local(0f);
                        k.If((c0 + kc < end) & (d0 + e < dim), () => value.V = keys[(first + c0 + kc) * dim + d0 + e]);
                        staged[kc * Stride + e] = value.V;
                    });
                    k.Barrier();
                    var sum = score.V;
                    for (int e = 0; e < Depth; e++)
                    {
                        sum = k.Fma(queries[r * Stride + e], staged[c * Stride + e], sum);
                    }

                    score.V = sum;
                    k.Barrier();                                              // the stage is read before the next is written
                });

                var seen = (c0 + c < hi) & (c0 + c >= lo);
                var scaled = Capped(k, score.V * scale, softcap);
                scores[tid] = k.Select(seen, scaled, k.Float(float.NegativeInfinity));
                k.Barrier();

                // 2. Each row's max over the tile, and how much its earlier sums shrink (1 while the row has seen nothing).
                k.If(tid < rows, () =>
                {
                    var tileMax = scores[tid * tile];
                    for (int i = 1; i < tile; i++)
                    {
                        tileMax = k.Max(tileMax, scores[tid * tile + i]);
                    }

                    var old = maxes[tid];
                    var updated = k.Max(old, tileMax);
                    alphas[tid] = k.Select(updated > float.NegativeInfinity, k.Exp(old - updated), k.Float(1f));
                    maxes[tid] = updated;
                });
                k.Barrier();

                // 3. The weights, the totals and the rescaled outputs.
                var weight = k.Local(0f);
                k.If(seen, () => weight.V = k.Exp(scaled - maxes[r]));
                scores[tid] = weight.V;
                k.Barrier();
                var alpha = alphas[r];
                k.If(tid < rows, () =>
                {
                    var total = totals[tid] * alphas[tid];
                    for (int i = 0; i < tile; i++)
                    {
                        total = total + scores[tid * tile + i];
                    }

                    totals[tid] = total;
                });

                for (int j = 0; j < outputs; j++)
                {
                    acc[j].V = acc[j].V * alpha;
                }

                // 4. The weighted values, a stage of 64 dimensions at a time.
                var positions = k.Min(end - c0, k.Int(tile));
                for (int s = 0; s < stages; s++)
                {
                    int stage = s;
                    k.If(k.Int(stage * Depth) < dim, () =>
                    {
                        k.For(tid, k.Int(tile * Depth), width, i =>
                        {
                            var (vc, e) = (i / Depth, i % Depth);
                            var value = k.Local(0f);
                            k.If((vc < positions) & (stage * Depth + e < dim), () => value.V = values[(first + c0 + vc) * dim + stage * Depth + e]);
                            staged[vc * Depth + e] = value.V;
                        });
                        k.Barrier();
                        var sums = new Var[perStage];
                        for (int i = 0; i < perStage; i++)
                        {
                            sums[i] = acc[stage * perStage + i];
                        }

                        k.For(k.Int(0), positions, 1, vc =>
                        {
                            var p = scores[r * tile + vc];
                            for (int i = 0; i < perStage; i++)
                            {
                                sums[i].V = k.Fma(p, staged[vc * Depth + c + i * tile], sums[i].V);
                            }
                        });
                        k.Barrier();                                          // the stage (and the weights) read before reuse
                    });
                }
            });

            // A barrier before the totals are read: with no tile at all (every row of the block sees nothing), the
            // resets above are the last writes.
            k.Barrier();

            // The block's rows: outputs over totals, zeros for a row that saw nothing.
            k.If(r0 + r < rowCount, () =>
            {
                var row = queryBase + r0 + r;
                var total = totals[r];
                var any = total > 0f;
                for (int j = 0; j < outputs; j++)
                {
                    var (dd, value) = (j / perStage * Depth + c + j % perStage * tile, acc[j]);
                    k.If(dd < dim, () => y[row * dim + dd] = k.Select(any, value.V / total, k.Float(0f)));
                }

                if (lse is { } output)
                {
                    k.If(c.Eq(0), () => output[row] = k.Select(any, maxes[r] + k.Log(total), k.Float(float.NegativeInfinity)));
                }
            });
            k.Barrier();                                                      // max and total read before the next block resets them
        }, k.GroupsX);
        return k.Build();
    }

    // Δ[row] = Σ_d dOutput[row, d] · output[row, d], an invocation per row (every head's rows: [heads · rows, dim]).
    private static SpirvKernel SpanDelta()
    {
        var k = new KernelBuilder("attention_spans_delta", Block);
        var (output, dOutput, delta) = (k.Buffer("output"), k.Buffer("dOutput"), k.Buffer("delta"));
        var (rows, dim) = (k.PushInt("rows"), k.PushInt("dim"));
        Grid(k, rows, row =>
        {
            var sum = k.Local(0f);
            var o = row * dim;
            k.For(k.Int(0), dim, 1, d => sum.V = k.Fma(dOutput[o + d], output[o + d], sum.V));
            delta[row] = sum.V;
        });
        return k.Build();
    }

    // Σ_d a[row r, d] · b[row c, d] for invocation (r, c), the head staged TiledAttentionDepth dimensions at a time: rowsA
    // rows of `a` (row i at element aAt(i), zeros where aIn(i) does not hold) into `sa` [rowsA · (Depth + 1)], rowsB rows
    // of `b` into `sb` likewise, a barrier, the stage's products, a barrier (the stage read before the next is written).
    private static Val StagedDot(KernelBuilder k, Val tid, int width, Val dim, SharedArray sa, int rowsA, Func<Val, Val> aAt, Func<Val, Val> aIn, Buf a,
        SharedArray sb, int rowsB, Func<Val, Val> bAt, Func<Val, Val> bIn, Buf b, Val r, Val c)
    {
        const int Depth = TiledAttentionDepth, Stride = Depth + 1;
        var sum = k.Local(0f);
        k.For(k.Int(0), dim, Depth, d0 =>
        {
            k.For(tid, k.Int(rowsA * Depth), width, i =>
            {
                var (row, e) = (i / Depth, i % Depth);
                var value = k.Local(0f);
                k.If(aIn(row) & (d0 + e < dim), () => value.V = a[aAt(row) + d0 + e]);
                sa[row * Stride + e] = value.V;
            });
            k.For(tid, k.Int(rowsB * Depth), width, i =>
            {
                var (row, e) = (i / Depth, i % Depth);
                var value = k.Local(0f);
                k.If(bIn(row) & (d0 + e < dim), () => value.V = b[bAt(row) + d0 + e]);
                sb[row * Stride + e] = value.V;
            });
            k.Barrier();
            var total = sum.V;
            for (int e = 0; e < Depth; e++)
            {
                total = k.Fma(sa[r * Stride + e], sb[c * Stride + e], total);
            }

            sum.V = total;
            k.Barrier();
        });
        return sum.V;
    }

    // The gradient for the queries: a workgroup of W invocations takes R = TiledAttentionRows(W) rows of one head (a
    // block, as attention_spans) and goes through the keys C = W / R at a time from the smallest start of its rows to their
    // largest end. Invocation (r, c) = (t / C, t % C), per tile:
    //   1. s = q_r · k_c and dp = dOutput_r · v_c, each staged 64 dimensions at a time (StagedDot);
    //   2. where row r's range holds key c: p = exp(cap(scale · s) - lse_r), ds = p · (dp - Δ_r) · cap'(s) (else 0), into
    //      slopes[t];
    //   3. the tile's keys, 64 dimensions at a time, are staged and each invocation adds Σ_c ds[r, c] · k[c, d] to the
    //      dimensions it keeps (d = lane + j·C, as attention_spans keeps its outputs), keys in order.
    // At the end dq[r] += scale · those sums. Workgroup memory: queries [R · 65] and staged [C · 65] written after the
    // barrier that ends their previous reads, read after the next; slopes [W] written by invocation t only, read by its
    // row's invocations after the first barrier of step 3, rewritten only after the last.
    private static SpirvKernel SpanBackwardQueries()
    {
        int width = Block, rows = TiledAttentionRows(width), tile = TiledAttentionPositions(width);
        const int Depth = TiledAttentionDepth;
        int outputs = AttentionMaxDim / tile, perStage = Depth / tile, stages = AttentionMaxDim / Depth;
        var k = new KernelBuilder("attention_spans_backward_dq", width);
        var (q, keys, values, starts, ends) = (k.Buffer("q"), k.Buffer("keys"), k.Buffer("values"), k.Buffer("starts"), k.Buffer("ends"));
        var (lse, dOutput, delta, dq) = (k.Buffer("lse"), k.Buffer("dOutput"), k.Buffer("delta"), k.Buffer("dq"));
        var (heads, rowCount, keyRows, dim) = (k.PushInt("heads"), k.PushInt("rows"), k.PushInt("keyRows"), k.PushInt("dim"));
        var (group, headsPerTable) = (k.PushInt("group"), k.PushInt("headsPerTable"));
        var (scale, softcap) = (k.PushFloat("scale"), k.PushFloat("softcap"));
        var queries = k.Shared("queries", rows * (Depth + 1));
        var staged = k.Shared("staged", tile * (Depth + 1));
        var slopes = k.Shared("slopes", width);
        var tid = k.LocalX;
        var (r, c) = (tid / tile, tid % tile);
        var blocksPerHead = (rowCount + (rows - 1)) / rows;
        var acc = new Var[outputs];
        for (int j = 0; j < outputs; j++)
        {
            acc[j] = k.Local(ScalarKind.Float);
        }

        k.For(k.GroupX, heads * blocksPerHead, block =>
        {
            var h = block / blocksPerHead;
            var r0 = block % blocksPerHead * rows;
            var first = h / group * keyRows;                                  // the key/value head's first row
            var table = h / headsPerTable * rowCount;                         // the head's table of ranges
            var queryBase = h * rowCount;
            var mine = k.Min(r0 + r, rowCount - 1);                            // this invocation's row within the head
            var lo = k.Clamp(starts[table + mine].ToInt(), k.Int(0), keyRows);
            var hi = k.Clamp(ends[table + mine].ToInt(), k.Int(0), keyRows);
            var rowLse = lse[queryBase + mine];
            var rowDelta = delta[queryBase + mine];

            // The keys the block sees: from the smallest start of its rows that see any to their largest end.
            var from = k.Local(keyRows);
            var count = k.Local(0);
            k.For(r0, k.Min(r0 + rows, rowCount), 1, row =>
            {
                var s = k.Clamp(starts[table + row].ToInt(), k.Int(0), keyRows);
                var e = k.Clamp(ends[table + row].ToInt(), k.Int(0), keyRows);
                k.If(e > s, () =>
                {
                    from.V = k.Min(from.V, s);
                    count.V = k.Max(count.V, e);
                });
            });

            foreach (var v in acc)
            {
                v.V = k.Float(0f);
            }

            var end = count.V;
            k.For(from.V, end, tile, c0 =>
            {
                Val RowAt(Val row) => (queryBase + k.Min(r0 + row, rowCount - 1)) * dim;
                Val KeyAt(Val key) => (first + c0 + key) * dim;
                Val Always(Val row) => k.Bool(true);
                Val InTile(Val key) => c0 + key < end;
                var score = StagedDot(k, tid, width, dim, queries, rows, RowAt, Always, q, staged, tile, KeyAt, InTile, keys, r, c);
                var dp = StagedDot(k, tid, width, dim, queries, rows, RowAt, Always, dOutput, staged, tile, KeyAt, InTile, values, r, c);
                var seen = (c0 + c < hi) & (c0 + c >= lo);
                var ds = k.Local(0f);
                k.If(seen, () =>
                {
                    var capped = Capped(k, score * scale, softcap);
                    ds.V = k.Exp(capped - rowLse) * (dp - rowDelta) * CapSlope(k, capped, softcap);
                });
                slopes[tid] = ds.V;

                // Σ_c ds[r, c] · keys[c, d], the tile's keys staged 64 dimensions at a time.
                var positions = k.Min(end - c0, k.Int(tile));
                for (int s = 0; s < stages; s++)
                {
                    int stage = s;
                    k.If(k.Int(stage * Depth) < dim, () =>
                    {
                        k.For(tid, k.Int(tile * Depth), width, i =>
                        {
                            var (kc, e) = (i / Depth, i % Depth);
                            var value = k.Local(0f);
                            k.If((kc < positions) & (stage * Depth + e < dim), () => value.V = keys[(first + c0 + kc) * dim + stage * Depth + e]);
                            staged[kc * Depth + e] = value.V;
                        });
                        k.Barrier();
                        k.For(k.Int(0), positions, 1, kc =>
                        {
                            var weight = slopes[r * tile + kc];
                            for (int i = 0; i < perStage; i++)
                            {
                                var sum = acc[stage * perStage + i];
                                sum.V = k.Fma(weight, staged[kc * Depth + c + i * tile], sum.V);
                            }
                        });
                        k.Barrier();                                          // the stage (and the slopes) read before reuse
                    });
                }
            });

            k.If(r0 + r < rowCount, () =>
            {
                var row = queryBase + r0 + r;
                for (int j = 0; j < outputs; j++)
                {
                    var (dd, value) = (j / perStage * Depth + c + j % perStage * tile, acc[j]);
                    k.If(dd < dim, () =>
                    {
                        var at = row * dim + dd;
                        dq[at] = k.Fma(scale, value.V, dq[at]);
                    });
                }
            });
        }, k.GroupsX);
        return k.Build();
    }

    // The gradient for the keys and values: a workgroup of W invocations takes K = TiledAttentionRows(W) keys of one
    // key/value head (a block) and goes through every query head of its group, each head's rows C = W / K at a time (a
    // tile); a tile none of whose rows sees a key of the block is skipped (every invocation reads the tile's ranges, so
    // all take the same branch). Invocation (r, c) = (t / C, t % C), per tile:
    //   1. s = k_r · q_c and dp = v_r · dOutput_c, staged 64 dimensions at a time (StagedDot);
    //   2. where row c's range holds key r: p = exp(cap(scale · s) - lse_c), ds = p · (dp - Δ_c) · cap'(s) (else 0);
    //   3. p into weights[t], the tile's dOutput rows staged 64 dimensions at a time, and each invocation adds
    //      Σ_c p[r, c] · dOutput[c, d] to the dimensions of key r it keeps (d = lane + j·C), rows in order; then ds into
    //      weights[t] and the tile's query rows likewise for Σ_c ds[r, c] · q[c, d].
    // At the end dkeys[r] += scale · the second sums and dvalues[r] += the first. Workgroup memory: keyStage [K · 65]
    // and rowStage [C · 65] as StagedDot's; weights [W] written by invocation t only, after the barrier ending its
    // previous reads, and read after the next.
    private static SpirvKernel SpanBackwardKeys()
    {
        int width = Block, keysPerBlock = TiledAttentionRows(width), tile = TiledAttentionPositions(width);
        const int Depth = TiledAttentionDepth;
        int outputs = AttentionMaxDim / tile, perStage = Depth / tile, stages = AttentionMaxDim / Depth;
        var k = new KernelBuilder("attention_spans_backward_dkv", width);
        var (q, keys, values, starts, ends) = (k.Buffer("q"), k.Buffer("keys"), k.Buffer("values"), k.Buffer("starts"), k.Buffer("ends"));
        var (lse, dOutput, delta, dkeys, dvalues) = (k.Buffer("lse"), k.Buffer("dOutput"), k.Buffer("delta"), k.Buffer("dkeys"), k.Buffer("dvalues"));
        var (heads, rowCount, keyRows, dim) = (k.PushInt("heads"), k.PushInt("rows"), k.PushInt("keyRows"), k.PushInt("dim"));
        var (group, headsPerTable) = (k.PushInt("group"), k.PushInt("headsPerTable"));
        var (scale, softcap) = (k.PushFloat("scale"), k.PushFloat("softcap"));
        var keyStage = k.Shared("keyStage", keysPerBlock * (Depth + 1));
        var rowStage = k.Shared("rowStage", tile * (Depth + 1));
        var weights = k.Shared("weights", width);
        var tid = k.LocalX;
        var (r, c) = (tid / tile, tid % tile);
        var keyBlocks = (keyRows + (keysPerBlock - 1)) / keysPerBlock;
        var kvHeads = heads / group;
        var dk = new Var[outputs];
        var dv = new Var[outputs];
        for (int j = 0; j < outputs; j++)
        {
            (dk[j], dv[j]) = (k.Local(ScalarKind.Float), k.Local(ScalarKind.Float));
        }

        k.For(k.GroupX, kvHeads * keyBlocks, block =>
        {
            var g = block / keyBlocks;
            var k0 = block % keyBlocks * keysPerBlock;
            var first = g * keyRows;                                          // the key/value head's first row
            var key = k0 + r;                                                 // this invocation's key
            var keyEnd = k.Min(k0 + keysPerBlock, keyRows);
            for (int j = 0; j < outputs; j++)
            {
                dk[j].V = k.Float(0f);
                dv[j].V = k.Float(0f);
            }

            k.For(g * group, g * group + group, 1, h =>
            {
                var table = h / headsPerTable * rowCount;
                var queryBase = h * rowCount;
                k.For(k.Int(0), rowCount, tile, i0 =>
                {
                    var count = k.Min(rowCount - i0, k.Int(tile));
                    var any = k.Local(0);
                    k.For(i0, i0 + count, 1, row =>
                    {
                        var s = k.Clamp(starts[table + row].ToInt(), k.Int(0), keyRows);
                        var e = k.Clamp(ends[table + row].ToInt(), k.Int(0), keyRows);
                        k.If((e > s) & (s < keyEnd) & (e > k0), () => any.V = k.Int(1));
                    });
                    k.If(any.V > 0, () =>
                    {
                        var mine = k.Min(i0 + c, rowCount - 1);                // this invocation's row within the head
                        var lo = k.Clamp(starts[table + mine].ToInt(), k.Int(0), keyRows);
                        var hi = k.Clamp(ends[table + mine].ToInt(), k.Int(0), keyRows);
                        Val KeyAt(Val row) => (first + k.Min(k0 + row, keyRows - 1)) * dim;
                        Val KeyIn(Val row) => k0 + row < keyRows;
                        Val RowAt(Val row) => (queryBase + k.Min(i0 + row, rowCount - 1)) * dim;
                        Val RowIn(Val row) => row < count;
                        var score = StagedDot(k, tid, width, dim, keyStage, keysPerBlock, KeyAt, KeyIn, keys, rowStage, tile, RowAt, RowIn, q, r, c);
                        var dp = StagedDot(k, tid, width, dim, keyStage, keysPerBlock, KeyAt, KeyIn, values, rowStage, tile, RowAt, RowIn, dOutput, r, c);
                        var seen = (c < count) & (key < hi) & (key >= lo);
                        var (p, ds) = (k.Local(0f), k.Local(0f));
                        k.If(seen, () =>
                        {
                            var capped = Capped(k, score * scale, softcap);
                            var weight = k.Exp(capped - lse[queryBase + mine]);
                            p.V = weight;
                            ds.V = weight * (dp - delta[queryBase + mine]) * CapSlope(k, capped, softcap);
                        });

                        // Σ_c weight[r, c] · source[c, d] into sums, the tile's rows of `source` staged 64 dimensions at a time.
                        void Accumulate(Val weight, Buf source, Var[] sums)
                        {
                            weights[tid] = weight;
                            for (int s = 0; s < stages; s++)
                            {
                                int stage = s;
                                k.If(k.Int(stage * Depth) < dim, () =>
                                {
                                    k.For(tid, k.Int(tile * Depth), width, i =>
                                    {
                                        var (rc, e) = (i / Depth, i % Depth);
                                        var value = k.Local(0f);
                                        k.If((rc < count) & (stage * Depth + e < dim), () => value.V = source[(queryBase + i0 + rc) * dim + stage * Depth + e]);
                                        rowStage[rc * Depth + e] = value.V;
                                    });
                                    k.Barrier();
                                    k.For(k.Int(0), count, 1, rc =>
                                    {
                                        var w = weights[r * tile + rc];
                                        for (int i = 0; i < perStage; i++)
                                        {
                                            var sum = sums[stage * perStage + i];
                                            sum.V = k.Fma(w, rowStage[rc * Depth + c + i * tile], sum.V);
                                        }
                                    });
                                    k.Barrier();                              // the stage (and the weights) read before reuse
                                });
                            }
                        }

                        Accumulate(p.V, dOutput, dv);
                        Accumulate(ds.V, q, dk);
                    });
                });
            });

            k.If(key < keyRows, () =>
            {
                for (int j = 0; j < outputs; j++)
                {
                    var (dd, gk, gv) = (j / perStage * Depth + c + j % perStage * tile, dk[j], dv[j]);
                    k.If(dd < dim, () =>
                    {
                        var at = (first + key) * dim + dd;
                        dkeys[at] = k.Fma(scale, gk.V, dkeys[at]);
                        dvalues[at] = dvalues[at] + gv.V;
                    });
                }
            });
        }, k.GroupsX);
        return k.Build();
    }
}
