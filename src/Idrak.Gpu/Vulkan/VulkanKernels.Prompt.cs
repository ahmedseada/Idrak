// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Vulkan;

// Kernels for prompts (many query rows at once): tiled attention over a key/value cache, with and without each row's
// log-sum-exp, and the packed weight products (int8, int4, bfloat16) tiled like batched_matmul, the weights expanded as
// they are staged instead of into a float copy first.
internal static partial class VulkanKernels
{
    private static IEnumerable<(string, Func<SpirvKernel>)> PromptKernels()
    {
        yield return ("attention_tiled", () => TiledAttention("attention_tiled", logSumExp: false));
        yield return ("attention_tiled_lse", () => TiledAttention("attention_tiled_lse", logSumExp: true));
        foreach (var format in new[] { PackedFormat.Int8, PackedFormat.Int4, PackedFormat.BFloat16 })
        {
            string name = PackedGemmName(format);
            yield return (name, () => PackedGemm(name, format));
        }
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
    private static SpirvKernel TiledAttention(string name, bool logSumExp)
    {
        int width = Block, rows = TiledAttentionRows(width), tile = TiledAttentionPositions(width);
        const int Depth = TiledAttentionDepth, Stride = Depth + 1;
        int outputs = AttentionMaxDim / tile, perStage = Depth / tile, stages = AttentionMaxDim / Depth;
        var k = new KernelBuilder(name, width);
        var (q, keys, values, position, y) = (k.Buffer("q"), k.Buffer("keys"), k.Buffer("values"), k.Buffer("position"), k.Buffer("y"));
        var lse = logSumExp ? k.Buffer("logSumExp") : null;
        var (heads, rowsPerHead, steps, capacity, dim) = (k.PushInt("heads"), k.PushInt("rowsPerHead"), k.PushInt("steps"), k.PushInt("capacity"), k.PushInt("dim"));
        var scale = k.PushFloat("scale");
        var (window, softcap) = (k.PushInt("window"), k.PushFloat("softcap"));
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
            var lo = WindowStart(k, limit, window);                            // the first of them
            // Positions the block sees: its rows' largest step (the last row's, unless the block wraps past a step), from
            // its smallest step's window start (the first row's, unless the block wraps).
            var last = k.Min(r0 + rows, rowsPerHead) - 1;
            var wraps = (last / steps).Ne(r0 / steps);
            var lastStep = k.Select(wraps, steps - 1, last % steps);
            var count = k.Min(start + lastStep, capacity - 1) + 1;
            var firstStep = k.Select(wraps, k.Int(0), r0 % steps);
            var from = WindowStart(k, k.Min(start + firstStep, capacity - 1) + 1, window);
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
                        k.If((c0 + kc < count) & (d0 + e < dim), () => value.V = keys[(first + c0 + kc) * dim + d0 + e]);
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

            // The block's rows: outputs over totals (written before the last tile's final barrier).
            k.If(r0 + r < rowsPerHead, () =>
            {
                var row = queryBase + r0 + r;
                var total = totals[r];
                for (int j = 0; j < outputs; j++)
                {
                    var (dd, value) = (j / perStage * Depth + c + j % perStage * tile, acc[j]);
                    k.If(dd < dim, () => y[row * dim + dd] = value.V / total);
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

    /// <summary>The prompt-sized packed product kernel of a format.</summary>
    public static string PackedGemmName(PackedFormat format) => format switch
    {
        PackedFormat.Int8 => "int8_gemm",
        PackedFormat.Int4 => "int4_gemm",
        _ => "bf16_gemm",
    };

    /// <summary>Rows and columns of y one workgroup of the packed product computes at a width (batched_matmul's block).</summary>
    public static int PackedGemmEdge(int width) => MatPer * MatSide(width);

    // y[m, n] = x[m, k] · w(k, n) for packed weights (layouts as PackedGemv's), many rows: batched_matmul's register-blocked
    // tiling (S × S invocations, S = MatSide(width), each keeping 4 × 4 outputs of a T × T block of y, T = 4S; grid x =
    // ⌈n / T⌉ column blocks, y = ⌈m / T⌉ row blocks), with the 16 × T tile of w staged from its packed words: invocation
    // l takes word l of the tile (16 rows of k × T / (columns per word) words; T is a multiple of every format's columns
    // per word, so a block starts on a word) and writes its columns as floats (int4 times the scale of their group of 32
    // rows; int8's per-column scales multiply the sums at the end). Zeros past k, n or m. Every output adds its k terms
    // in order, as batched_matmul does.
    //
    // Workgroup memory: tileA[kk · (T + 1) + r] and tileB[kk · (T + 1) + col] are written only between the barrier ending
    // the previous step's reads and the barrier before this step's reads, each element by exactly one invocation.
    private static SpirvKernel PackedGemm(string name, PackedFormat format)
    {
        int side = MatSide(Block), threads = side * side, Per = MatPer, tileEdge = Per * side, stride = tileEdge + 1;
        int sideShift = System.Numerics.BitOperations.Log2((uint)side);
        int perWord = ColumnsPerWord(format), tileWords = tileEdge / perWord, wordLoads = MatDepth * tileWords;
        var k = new KernelBuilder(name, threads);
        var x = k.Buffer("x");
        var q = k.Buffer(format == PackedFormat.BFloat16 ? "packed" : "q");
        var scales = format == PackedFormat.BFloat16 ? null : k.Buffer("scales");
        var y = k.Buffer("y");
        var (m, n, depth) = (k.PushInt("m"), k.PushInt("n"), k.PushInt("k"));
        var tileA = k.Shared("tileA", MatDepth * stride);
        var tileB = k.Shared("tileB", MatDepth * stride);
        var tid = k.LocalX;
        var (tx, ty) = (tid & (side - 1), tid >> sideShift);
        var rowBase = k.GroupY * tileEdge;
        var colBase = k.GroupX * tileEdge;
        var words = (n + (perWord - 1)) / perWord;
        var wordBase = k.GroupX * tileWords;
        var acc = new Var[Per * Per];
        for (int i = 0; i < acc.Length; i++)
        {
            acc[i] = k.Local(0f);
        }

        k.For(k.Int(0), depth, MatDepth, t =>
        {
            // x[rowBase + r, t + ka]: 16 consecutive invocations along k.
            for (int p = 0; p < tileEdge * MatDepth / threads; p++)
            {
                var l = tid + p * threads;
                var (r, ka) = (l >> 4, l & (MatDepth - 1));
                var (row, kk) = (rowBase + r, t + ka);
                var value = k.Local(0f);
                k.If((row < m) & (kk < depth), () => value.V = x[row * depth + kk]);
                tileA[ka * stride + r] = value.V;
            }

            // The tile's packed words: consecutive invocations along a row of k.
            for (int p = 0; p * threads < wordLoads; p++)
            {
                var l = tid + p * threads;
                void Load()
                {
                    var (kb, wv) = (l / tileWords, l % tileWords);
                    var kk = t + kb;
                    var bits = k.Local(k.Int(0));
                    k.If((kk < depth) & (wordBase + wv < words), () => bits.V = q.Int(kk * words + wordBase + wv));
                    var b = bits.V;
                    for (int ci = 0; ci < perWord; ci++)
                    {
                        Val w = format switch
                        {
                            PackedFormat.Int8 => ((b << (24 - 8 * ci)) >> 24).ToFloat(),
                            PackedFormat.Int4 => ((b << (28 - 4 * ci)) >> 28).ToFloat()
                                * scales![k.Min(kk, depth - 1) / 32 * (words * 8) + k.Min(wordBase + wv, words - 1) * 8 + ci],
                            _ => ci == 0 ? (b.AsUInt() << 16).AsFloat() : (b.AsUInt() & k.UInt(0xFFFF0000u)).AsFloat(),
                        };
                        tileB[kb * stride + wv * perWord + ci] = w;
                    }
                }

                if ((p + 1) * threads <= wordLoads)
                {
                    Load();
                }
                else
                {
                    k.If(l < wordLoads, Load);
                }
            }

            k.Barrier();
            var sums = acc.Select(v => v.V).ToArray();
            for (int e = 0; e < MatDepth; e++)
            {
                var av = new Val[Per];
                var bv = new Val[Per];
                for (int i = 0; i < Per; i++)
                {
                    av[i] = tileA[k.Int(e * stride + side * i) + ty];
                    bv[i] = tileB[k.Int(e * stride + side * i) + tx];
                }

                for (int i = 0; i < Per; i++)
                {
                    for (int j = 0; j < Per; j++)
                    {
                        sums[i * Per + j] = k.Fma(av[i], bv[j], sums[i * Per + j]);
                    }
                }
            }

            for (int i = 0; i < acc.Length; i++)
            {
                acc[i].V = sums[i];
            }

            k.Barrier();
        });

        for (int i = 0; i < Per; i++)
        {
            for (int j = 0; j < Per; j++)
            {
                var (row, col, value) = (rowBase + ty + side * i, colBase + tx + side * j, acc[i * Per + j]);
                k.If((row < m) & (col < n), () => y[row * n + col] = scales is not null && format == PackedFormat.Int8 ? value.V * scales[col] : value.V);
            }
        }

        return k.Build();
    }
}
