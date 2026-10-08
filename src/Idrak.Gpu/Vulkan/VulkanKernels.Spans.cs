// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Vulkan;

// Attention over one range of keys per query row (Backend.AttentionSpans), tiled as attention_tiled: the same
// workgroups, staging and online softmax, with each row's range read from the starts and ends buffers instead of
// computed from a causal limit and a window.
internal static partial class VulkanKernels
{
    private static IEnumerable<(string, Func<SpirvKernel>)> SpanKernels()
    {
        yield return ("attention_spans", () => SpanAttention("attention_spans", logSumExp: false));
        yield return ("attention_spans_lse", () => SpanAttention("attention_spans_lse", logSumExp: true));
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
}
