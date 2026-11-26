// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Vulkan;

// Kernels for prompts (many query rows at once): tiled attention over a key/value cache, with and without each row's
// log-sum-exp.
internal static partial class VulkanKernels
{
    private static IEnumerable<(string, Func<SpirvKernel>)> PromptKernels()
    {
        yield return ("attention_tiled", () => TiledAttention("attention_tiled", logSumExp: false));
        yield return ("attention_tiled_lse", () => TiledAttention("attention_tiled_lse", logSumExp: true));

        // attention_tiled for rows of different lengths (Backend.AttentionRows): each row from its own start.
        yield return ("attention_tiled_rows_f32", () => TiledAttention("attention_tiled_rows_f32", logSumExp: false, rowStarts: true));
    }

    /// <summary>Query rows one workgroup of the tiled attention takes at a width (the side of batched_matmul's square).</summary>
    public static int TiledAttentionRows(int width) => MatSide(width);

    /// <summary>Cached positions one tile of the tiled attention takes at a width: the rest of the workgroup.</summary>
    public static int TiledAttentionPositions(int width) => width / MatSide(width);

    // Dimensions of the head staged per step of the tiled attention (a multiple of every TiledAttentionPositions).
    private const int TiledAttentionDepth = 64;

    // Attention of query rows q [heads, rowsPerHead, dim] over keys and values [heads, capacity, dim], as attention_decode
    // (row i of head h sees positions c ≤ min(position[0] + i % steps, capacity - 1), from its window's start with a
    // window, its scores soft-capped with a cap), for many rows: a workgroup of W
    // invocations takes R = TiledAttentionRows(W) consecutive rows of one head (a block) and goes through the positions
    // C = W / R at a time (a tile), so each key and value it reads serves R rows (attention_decode reads them once per
    // row). Invocation t is (r, c) = (t / C, t % C) for the scores and (r, lane = t % C) for the outputs, which it keeps in
    // registers: dimensions lane + j·C, j < 256 / C. Per tile, with an online softmax per row:
    //   1. the score of (r, c): q_r · k_c over the head, staged 64 dimensions at a time (the block's queries and the
    //      tile's keys, zeros past dim or past the block's last position), then times scale and soft-capped (-∞ outside
    //      row r's window and past its limit);
    //   2. invocation r < R takes the tile's largest score of row r: max' = max(max, tile max), alpha = exp(max - max');
    //   3. invocation (r, c) stores p = exp(score - max') (0 past the limit); invocation r < R updates its row's total,
    //      total · alpha + Σ p, and every invocation rescales its outputs by its row's alpha;
    //   4. the tile's values, 64 dimensions at a time, are staged and each output adds Σ_c p[r, c] · v[c, d].
    // At the end y = outputs / total (and the log-sum-exp max + log total). Rows past rowsPerHead repeat the last row of
    // the head and store nothing. The tiles start at the block's lowest window start (0 without a window). Without a
    // window every row sees position 0, so its max is finite after the first tile; with one, a row may see nothing in the
    // block's first tiles, and its max stays -∞ (alpha 1, its sums 0) until its window begins.
    //
    // Workgroup memory — every phase is separated by a barrier:
    //   queries[R · 65] and staged[C · 65] (keys, rows padded by one so (r, c) read different banks; then values, C · 64)
    //   written after the barrier that ends the previous reads of them, read after the next;
    //   scores[W]: written by invocation t only (scores, then weights), read by the row's invocations after a barrier;
    //   max[R], alpha[R], total[R]: written by invocation r < R only (max and alpha after a barrier ending the previous
    //   tile's reads of alpha; total read only at the end of the block), read after a barrier.
    //
    // With `rowStarts` (attention_tiled_rows_*: rows of different lengths), row i of head h also starts at its own
    // starts[(h / headsPerRow) · steps + i % steps] (a binding before the position, a push constant after softcap): the
    // tiles start at the lowest first position of the block's rows that see any (none: no tile), and a row that sees
    // nothing (padding) keeps a total of 0 and gets zeros.
    private static SpirvKernel TiledAttention(string name, bool logSumExp, bool rowStarts = false)
    {
        int width = Block, rows = TiledAttentionRows(width), tile = TiledAttentionPositions(width);
        const int Depth = TiledAttentionDepth, Stride = Depth + 1;
        int outputs = AttentionMaxDim / tile, perStage = Depth / tile, stages = AttentionMaxDim / Depth;
        var k = new KernelBuilder(name, width);
        var (q, keys, values) = (k.Buffer("q"), k.Buffer("keys"), k.Buffer("values"));
        var starts = rowStarts ? k.Buffer("starts") : null;
        var (position, y) = (k.Buffer("position"), k.Buffer("y"));
        var lse = logSumExp ? k.Buffer("logSumExp") : null;
        var (heads, rowsPerHead, steps, capacity, dim) = (k.PushInt("heads"), k.PushInt("rowsPerHead"), k.PushInt("steps"), k.PushInt("capacity"), k.PushInt("dim"));
        var scale = k.PushFloat("scale");
        var (window, softcap) = (k.PushInt("window"), k.PushFloat("softcap"));
        Val? headsPerRow = rowStarts ? k.PushInt("headsPerRow") : null;

        // Dimension d0 + e of cached row `slot` (= head · capacity + position).
        Val Cached(Buf cache, Val slot, Val d0, Val e) => cache[slot * dim + d0 + e];

        // The first position row `row` of head h sees (its window's start, its own start), given its limit.
        Val RowStart(Val h, Val row, Val limit)
        {
            var lo = WindowStart(k, limit, window);
            return starts is null ? lo : k.Max(lo, starts[h / headsPerRow!.Value * steps + row % steps].ToInt());
        }
        var queries = k.Shared("queries", rows * Stride);
        var staged = k.Shared("staged", tile * Stride);
        var scores = k.Shared("scores", width);
        var maxes = k.Shared("max", rows);
        var alphas = k.Shared("alpha", rows);
        var totals = k.Shared("total", rows);
        var tid = k.LocalX;
        var (r, c) = (tid / tile, tid % tile);
        var start = position[k.Int(0)].ToInt();
        var blocksPerHead = (rowsPerHead + (rows - 1)) / rows;
        var acc = new Var[outputs];
        for (int j = 0; j < outputs; j++)
        {
            acc[j] = k.Local(ScalarKind.Float);
        }

        k.For(k.GroupX, heads * blocksPerHead, block =>
        {
            var h = block / blocksPerHead;
            var r0 = block % blocksPerHead * rows;
            var first = h * capacity;                                         // the head's first cached row
            var mine = k.Min(r0 + r, rowsPerHead - 1);                         // this invocation's row within the head
            var limit = k.Min(start + mine % steps, capacity - 1) + 1;         // positions the row sees
            var lo = RowStart(h, mine, limit);                                 // the first of them
            // Positions the block sees: its rows' largest step (the last row's, unless the block wraps past a step), from
            // its smallest step's window start (the first row's, unless the block wraps); with rows' own starts, from the
            // lowest first position of its rows that see any (none: from = count, no tile).
            var last = k.Min(r0 + rows, rowsPerHead) - 1;
            var wraps = (last / steps).Ne(r0 / steps);
            var lastStep = k.Select(wraps, steps - 1, last % steps);
            var count = k.Min(start + lastStep, capacity - 1) + 1;
            Val from;
            if (starts is null)
            {
                var firstStep = k.Select(wraps, k.Int(0), r0 % steps);
                from = WindowStart(k, k.Min(start + firstStep, capacity - 1) + 1, window);
            }
            else
            {
                var least = k.Local(count);
                k.For(r0, last + 1, 1, row =>
                {
                    var rowLimit = k.Min(start + row % steps, capacity - 1) + 1;
                    var rowFirst = RowStart(h, row, rowLimit);
                    k.If(rowFirst < rowLimit, () => least.V = k.Min(least.V, rowFirst));
                });
                from = least.V;
            }

            var queryBase = h * rowsPerHead;
            foreach (var v in acc)
            {
                v.V = k.Float(0f);
            }

            k.If(tid < rows, () =>
            {
                maxes[tid] = k.Float(float.NegativeInfinity);
                totals[tid] = k.Float(0f);
            });

            k.For(from, count, tile, c0 =>
            {
                // 1. Scores, a stage of 64 dimensions at a time.
                var score = k.Local(0f);
                k.For(k.Int(0), dim, Depth, d0 =>
                {
                    k.For(tid, k.Int(rows * Depth), width, i =>
                    {
                        var (qr, e) = (i / Depth, i % Depth);
                        var value = k.Local(0f);
                        k.If(d0 + e < dim, () => value.V = q[(queryBase + k.Min(r0 + qr, rowsPerHead - 1)) * dim + d0 + e]);
                        queries[qr * Stride + e] = value.V;
                    });
                    k.For(tid, k.Int(tile * Depth), width, i =>
                    {
                        var (kc, e) = (i / Depth, i % Depth);
                        var value = k.Local(0f);
                        k.If((c0 + kc < count) & (d0 + e < dim), () => value.V = Cached(keys, first + c0 + kc, d0, e));
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

                var seen = (c0 + c < limit) & (c0 + c >= lo);
                var scaled = Capped(k, score.V * scale, softcap);
                scores[tid] = k.Select(seen, scaled, k.Float(float.NegativeInfinity));
                k.Barrier();

                // 2. Each row's max over the tile, and how much its earlier sums shrink.
                k.If(tid < rows, () =>
                {
                    var tileMax = scores[tid * tile];
                    for (int i = 1; i < tile; i++)
                    {
                        tileMax = k.Max(tileMax, scores[tid * tile + i]);
                    }

                    var old = maxes[tid];
                    var updated = k.Max(old, tileMax);
                    // 0 on the row's first tile with a position (old = -∞); 1 while it has none (updated = -∞ too).
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
                var positions = k.Min(count - c0, k.Int(tile));
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

            if (starts is not null)
            {
                k.Barrier();                                                  // no tile at all: the resets above are the last writes
            }

            // The block's rows: outputs over totals (written before the last tile's final barrier); with rows' own starts,
            // zeros for a row that saw nothing.
            k.If(r0 + r < rowsPerHead, () =>
            {
                var row = queryBase + r0 + r;
                var total = totals[r];
                for (int j = 0; j < outputs; j++)
                {
                    var (dd, value) = (j / perStage * Depth + c + j % perStage * tile, acc[j]);
                    k.If(dd < dim, () => y[row * dim + dd] = starts is null ? value.V / total : k.Select(total.Eq(k.Float(0f)), k.Float(0f), value.V / total));
                }

                if (lse is { } output)
                {
                    k.If(c.Eq(0), () => output[row] = maxes[r] + k.Log(total));
                }
            });
            k.Barrier();                                                      // max and total read before the next block resets them
        }, k.GroupsX);
        return k.Build();
    }
}
