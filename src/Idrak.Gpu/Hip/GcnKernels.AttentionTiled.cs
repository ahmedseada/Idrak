// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Gpu.Hip;

// The tiled attention over a float32 cache (plan 35, G6b): attention_tiled_r{R}_f32, R = 2, 4 or 8 query rows of one
// head a block, sharing each tile of 256 keys (rule 83's sharing: a key and a value row is read once for R rows, where
// attention_decode_f32 takes a block a row and reads them once a row). Each row's work is attention_decode_f32's, step
// for step: thread t scores key c0 + t against every row of the block (its dot products over the head in dimension
// order, four a step, the query rows from LDS), each row's tile maximum and weights' sum come from RowWriter's fixed tree
// (one array a row, the same shape), the row's running maximum, sum and accumulators are rescaled as there, and each
// thread adds the tile's weighted values for its dimension (d = tid) in key order, a value read once for the R rows.
// The tiles start at the block's first position and end at its last: with no window every row's range starts at 0, so
// its tiles are attention_decode_f32's and its result the same bits (a key past a row's range scores -∞ and weighs 0, as
// the threads past attention_decode_f32's tile do); with a window the rows' tiles may start elsewhere, a fixed order
// within the device-against-CPU tolerance. No splits: the tiled form serves many rows (the backend takes it where the
// row kernel takes one split). Heads up to AttentionTiledMaxDim (one dimension a thread).
//
// | Kernel                 | Arguments                                                                            | Launch (blocks of 256)              | Unit                 | Flops | Bytes |
// |------------------------|--------------------------------------------------------------------------------------|-------------------------------------|----------------------|-------|-------|
// | attention_tiled_rR_f32 | q, keys, values, pos, y, lse; rph, steps, cap, dim, scale, window, softcap, bph, n   | heads · ⌈rph / R⌉ (or fewer)        | row x position x dim | 4     | 8 / R |
//
// bph: a head's blocks, ⌈rph / R⌉; n = heads · bph. Block b takes rows [(b mod bph) · R, … + R) of head b / bph (those
// below rph); row i of head h sees positions [begin, end), end = min(pos + i mod steps, cap - 1) + 1, begin = end - window
// (0 without a window), as attention_decode_f32's rows.
internal static partial class GcnKernels
{
    /// <summary>The rows a tiled attention block's variants take (attention_tiled_r{R}_f32), the fewest first.</summary>
    internal static ReadOnlySpan<int> AttentionTiledRows => [2, 4, 8];

    /// <summary>The largest head size the tiled attention takes: one dimension a thread of the block.</summary>
    internal const int AttentionTiledMaxDim = RowThreads;

    /// <summary>The LDS a tiled attention block of <paramref name="rows"/> rows declares: the reductions' two areas (an array a row), the rows' tile weights and query rows.</summary>
    internal static int AttentionTiledLdsBytes(int rows) => 2 * 1024 * rows + 4 * AttentionTile * rows + 4 * AttentionTiledMaxDim * rows;

    /// <summary>The tiled attention kernel of <paramref name="rows"/> rows a block (2, 4 or 8).</summary>
    internal static string AttentionTiledKernel(int rows) => rows switch
    {
        2 => "attention_tiled_r2_f32",
        4 => "attention_tiled_r4_f32",
        8 => "attention_tiled_r8_f32",
        _ => throw new ArgumentOutOfRangeException(nameof(rows), rows, "the tiled attention takes 2, 4 or 8 rows a block"),
    };

    // Each row's maximum (RowWriter.Max combines the first array alone).
    private static string MaxOfEach(RowWriter writer, string[] held, string[] other)
    {
        var text = new System.Text.StringBuilder();
        for (int i = 0; i < held.Length; i++)
        {
            text.Append(CultureInfo.InvariantCulture, $"v_max_f32_e32 {held[i]}, {held[i]}, {other[i]}\n");
        }

        return text.ToString();
    }

    private static GcnKernelSource AttentionTiled(int rows)
    {
        var w = new RowWriter(AttentionTiledKernel(rows), arrays: rows, extraLds: 4 * AttentionTile * rows + 4 * AttentionTiledMaxDim * rows);
        int weightsLds = w.ExtraLdsBase, queryLds = w.ExtraLdsBase + 4 * AttentionTile * rows;
        var a = w.Arguments(["q", "keys", "values", "pos", "y", "lse"], ["rph", "steps", "cap", "dim", "scale", "window", "softcap", "bph", "n"]);
        string dim = a.Scalar("dim"), scale = a.Scalar("scale"), softcap = a.Scalar("softcap"), rph = a.Scalar("rph"), window = a.Scalar("window");
        string q = a.Pointer("q"), keys = a.Pointer("keys"), values = a.Pointer("values"), y = a.Pointer("y"), lse = a.Pointer("lse");

        // Once a call (rule 85): a cached row's bytes, ln 2, 1 / softcap, this thread's dimension, the device position.
        string rowBytes = w.S(), ln2 = w.S(), capInverse = w.V(), dimBytes = w.V(), tid4 = w.V(), bits = w.S(), value = w.V(), position = w.S(), last = w.S();
        string capRow = w.S(), lastDim = w.S(), lastDimBytes = w.V();
        w.L($"""
            s_lshl_b32 {rowBytes}, {dim}, 2
            s_mov_b32 {ln2}, {RowWriter.Ln2}
            v_rcp_f32_e32 {capInverse}, {softcap}
            v_lshlrev_b32_e32 {dimBytes}, 2, {w.Tid}
            v_lshlrev_b32_e32 {tid4}, 2, {w.Tid}
            s_load_dword {bits}, {a.Pointer("pos")}, 0x0
            v_cvt_u32_f32_e32 {value}, {bits}
            v_readfirstlane_b32 {position}, {value}
            s_sub_u32 {last}, {a.Scalar("cap")}, 1
            s_mul_i32 {capRow}, {a.Scalar("cap")}, {dim}
            s_sub_u32 {lastDim}, {dim}, 1
            v_min_u32_e32 {lastDimBytes}, {lastDim}, {w.Tid}
            v_lshlrev_b32_e32 {lastDimBytes}, 2, {lastDimBytes}
            """);
        string bySteps = AttentionReciprocal(w, a.Scalar("steps")), byBlocks = AttentionReciprocal(w, a.Scalar("bph"));
        w.BeginRows(a.Scalar("n"));

        // The block's rows: head, first row, each row's range [begin, end) (empty past rph) and output; the tiles' span.
        var (head, block) = AttentionDivMod(w, w.Row, a.Scalar("bph"), byBlocks);
        string first = w.S(), lo = w.S(), hi = w.S();
        w.L($"""
            s_mul_i32 {first}, {block}, {rows}
            s_mov_b32 {lo}, -1
            s_mov_b32 {hi}, 0
            """);
        var begins = new string[rows];
        var ends = new string[rows];
        var outputs = new string[rows];
        var rowIndex = new string[rows];
        for (int r = 0; r < rows; r++)
        {
            (begins[r], ends[r], rowIndex[r]) = (w.S(), w.S(), w.S());
            string i = w.S(), t = w.S();
            w.L($"s_add_u32 {i}, {first}, {r}");
            var (_, step) = AttentionDivMod(w, i, a.Scalar("steps"), bySteps);
            w.L($"""
                s_add_u32 {ends[r]}, {position}, {step}
                s_min_u32 {ends[r]}, {ends[r]}, {last}
                s_add_u32 {ends[r]}, {ends[r]}, 1
                s_sub_u32 {t}, {ends[r]}, {window}
                s_cmp_gt_u32 {ends[r]}, {window}
                s_cselect_b32 {begins[r]}, {t}, 0
                s_cmp_eq_u32 {window}, 0
                s_cselect_b32 {begins[r]}, 0, {begins[r]}
                s_cmp_lt_u32 {i}, {rph}
                s_cselect_b32 {ends[r]}, {ends[r]}, 0
                s_cselect_b32 {begins[r]}, {begins[r]}, 0
                s_cmp_lt_u32 {begins[r]}, {ends[r]}
                s_cselect_b32 {t}, {begins[r]}, -1
                s_min_u32 {lo}, {lo}, {t}
                s_max_u32 {hi}, {hi}, {ends[r]}
                s_mul_i32 {rowIndex[r]}, {head}, {rph}
                s_add_u32 {rowIndex[r]}, {rowIndex[r]}, {i}
                """);
            outputs[r] = w.ElementAddress(y, AttentionIndex(w, (rowIndex[r], dim)));
        }

        string cacheOffset = AttentionIndex(w, (head, capRow));
        string kb = w.ElementAddress(keys, cacheOffset), vb = w.ElementAddress(values, cacheOffset);

        // The block's query rows into LDS (a row 1,024 bytes apart; the rows past rph read the last row of the head:
        // their scores are masked).
        string inDim = w.SPair(), lastRow = w.S();
        w.L($"""
            v_cmp_gt_u32_e32 vcc, {dim}, {w.Tid}
            s_and_saveexec_b64 {inDim}, vcc
            s_sub_u32 {lastRow}, {rph}, 1
            """);
        string queryValue = w.V();
        w.L($"v_mov_b32_e32 {queryValue}, 0");
        for (int r = 0; r < rows; r++)
        {
            string i = w.S(), at = w.S();
            w.L($"""
                s_add_u32 {i}, {first}, {r}
                s_min_u32 {i}, {i}, {lastRow}
                s_mul_i32 {at}, {head}, {rph}
                s_add_u32 {at}, {at}, {i}
                s_mul_i32 {at}, {at}, {dim}
                """);
            string rowAt = w.ElementAddress(q, at);
            w.L($"""
                global_load_dword {queryValue}, {dimBytes}, {rowAt}
                ds_write_b32 {dimBytes}, {queryValue} offset:{queryLds + 1024 * r}
                """);
        }

        w.L($"s_or_b64 exec, exec, {inDim}");
        w.L($"s_barrier");

        // Each row's running maximum, sum and accumulator (this thread's dimension).
        var max = new string[rows];
        var sum = new string[rows];
        var acc = new string[rows];
        for (int r = 0; r < rows; r++)
        {
            (max[r], sum[r], acc[r]) = (w.V(), w.V(), w.V());
            w.L($"""
                v_mov_b32_e32 {max[r]}, {RowWriter.NegativeInfinity}
                v_mov_b32_e32 {sum[r]}, 0
                v_mov_b32_e32 {acc[r]}, 0
                """);
        }

        string c0 = w.S();
        w.L($"s_mov_b32 {c0}, {lo}");
        w.Label(".Lt_tile");
        string count = w.S(), lastKey = w.S(), key = w.V(), at4 = w.V(), queryRead = w.V(), d = w.S(), dim4 = w.S();
        var score = new string[rows];
        var dot = new string[rows];
        for (int r = 0; r < rows; r++)
        {
            (score[r], dot[r]) = (w.V(), w.V());
        }

        w.L($"""
            s_cmp_ge_u32 {c0}, {hi}
            s_cbranch_scc1 .Lt_tiles_end
            s_sub_u32 {count}, {hi}, {c0}
            s_min_u32 {count}, {count}, 0x{AttentionTile:x}
            """);
        for (int r = 0; r < rows; r++)
        {
            w.L($"v_mov_b32_e32 {score[r]}, {RowWriter.NegativeInfinity}");
        }

        // Phase 1: this thread's key c0 + tid against every row, the dot products in dimension order, four a step. Every
        // thread computes (the keys past the tile read its last key's row, then score -∞ as past every row's range), so
        // no register is written under a narrowed EXEC (one so written stays live from before: the block's occupancy).
        string clamped = w.V();
        w.L($"""
            s_sub_u32 {lastKey}, {count}, 1
            v_min_u32_e32 {clamped}, {lastKey}, {w.Tid}
            v_add_u32_e32 {clamped}, {c0}, {clamped}
            v_add_u32_e32 {key}, {c0}, {w.Tid}
            v_mul_lo_u32 {at4}, {clamped}, {rowBytes}
            v_mov_b32_e32 {queryRead}, 0x{queryLds:x}
            s_mov_b32 {d}, 0
            s_and_b32 {dim4}, {dim}, -4
            """);
        for (int r = 0; r < rows; r++)
        {
            w.L($"v_mov_b32_e32 {dot[r]}, 0");
        }

        w.Label(".Lt_dot");
        w.L($"s_cmp_ge_u32 {d}, {dim4}");
        w.L($"s_cbranch_scc1 .Lt_dot_end");
        var xs = new string[4];
        for (int e = 0; e < 4; e++)
        {
            xs[e] = w.V();
            w.L($"global_load_dword {xs[e]}, {at4}, {kb}{AttentionOffset(4 * e)}");
        }

        string queries = AttentionText($"%v[{AttentionRange(w, 4)}]");
        int qFirst = AttentionFirst(queries);
        for (int r = 0; r < rows; r++)
        {
            w.L($"ds_read_b128 {queries}, {queryRead} offset:{1024 * r}");
            for (int e = 0; e < 4; e++)
            {
                w.L($"v_fma_f32 {dot[r]}, %v{qFirst + e}, {xs[e]}, {dot[r]}");
            }
        }

        w.L($"""
            v_add_u32_e32 {at4}, 16, {at4}
            v_add_u32_e32 {queryRead}, 16, {queryRead}
            s_add_u32 {d}, {d}, 4
            s_branch .Lt_dot
            """);
        w.Label(".Lt_dot_end");
        string rest = w.S();
        w.L($"s_and_b32 {rest}, {dim}, 3");
        string qe = w.V();
        for (int e = 0; e < 3; e++)
        {
            string x = w.V();
            w.L($"s_cmp_le_u32 {rest}, {e}");
            w.L($"s_cbranch_scc1 .Lt_dot_tail");
            w.L($"global_load_dword {x}, {at4}, {kb}{AttentionOffset(4 * e)}");
            for (int r = 0; r < rows; r++)
            {
                w.L($"ds_read_b32 {qe}, {queryRead} offset:{1024 * r + 4 * e}");
                w.L($"v_fma_f32 {dot[r]}, {qe}, {x}, {dot[r]}");
            }
        }

        w.Label(".Lt_dot_tail");
        string capped = w.V(), seen = w.SPair(), below = w.SPair(), negInf = w.V();
        w.L($"v_mov_b32_e32 {negInf}, {RowWriter.NegativeInfinity}");
        for (int r = 0; r < rows; r++)
        {
            w.L($"v_mul_f32_e32 {score[r]}, {scale}, {dot[r]}");
        }

        w.L($"s_cmp_gt_i32 {softcap}, 0");
        w.L($"s_cbranch_scc0 .Lt_capped");
        for (int r = 0; r < rows; r++)
        {
            w.L($"""
                v_mul_f32_e32 {capped}, {score[r]}, {capInverse}
                v_mul_f32_e32 {capped}, 0x4038aa3b, {capped}
                v_exp_f32_e32 {capped}, {capped}
                v_add_f32_e32 {capped}, 1.0, {capped}
                v_rcp_f32_e32 {capped}, {capped}
                v_fma_f32 {capped}, {capped}, -2.0, 1.0
                v_mul_f32_e32 {score[r]}, {softcap}, {capped}
                """);
        }

        w.Label(".Lt_capped");
        for (int r = 0; r < rows; r++)
        {
            // A key outside the row's range scores -∞ (attention_decode_f32 leaves it out of the row's tile).
            w.L($"""
                v_cmp_le_u32_e64 {seen}, {begins[r]}, {key}
                v_cmp_gt_u32_e64 {below}, {ends[r]}, {key}
                s_and_b64 vcc, {seen}, {below}
                v_cndmask_b32_e32 {score[r]}, {negInf}, {score[r]}, vcc
                """);
        }


        // Phase 2: each row's tile maximum, the rescale of what came before, the weights (to LDS) and their sum.
        // (Each row's sum takes its maximum's register, free by then.)
        var best = new string[rows];
        var alpha = new string[rows];
        var shift = new string[rows];
        var top = new string[rows];
        var total = best;
        for (int r = 0; r < rows; r++)
        {
            (best[r], alpha[r], shift[r], top[r]) = (w.V(), w.V(), w.V(), w.V());
            w.L($"v_mov_b32_e32 {best[r]}, {score[r]}");
        }

        w.Reduce(MaxOfEach, best);
        string p = w.V(), tileAt = w.S();
        for (int r = 0; r < rows; r++)
        {
            w.L($"""
                v_max_f32_e32 {top[r]}, {max[r]}, {best[r]}
                v_cmp_eq_f32_e32 vcc, {RowWriter.NegativeInfinity}, {top[r]}
                v_cndmask_b32_e64 {shift[r]}, {top[r]}, 0, vcc
                v_sub_f32_e32 {alpha[r]}, {max[r]}, {shift[r]}
                v_mul_f32_e32 {alpha[r]}, {RowWriter.Log2E}, {alpha[r]}
                v_exp_f32_e32 {alpha[r]}, {alpha[r]}
                {RowWriter.Exp(p, score[r], shift[r])}
                ds_write_b32 {tid4}, {p} offset:{weightsLds + 1024 * r}
                v_mov_b32_e32 {total[r]}, {p}
                """);
        }

        w.Reduce(RowWriter.Add, total);
        for (int r = 0; r < rows; r++)
        {
            w.L($"""
                v_fma_f32 {sum[r]}, {sum[r]}, {alpha[r]}, {total[r]}
                v_mov_b32_e32 {max[r]}, {top[r]}
                """);
        }

        // Phase 3: this thread's dimension rescaled, then the tile's weighted values added in key order, four a step, a
        // value read once for every row (the threads past the head read its last dimension and store nothing).
        string o = w.V(), weightRead = w.V(), j = w.S(), count4 = w.S();
        w.L($"""
            s_mul_i32 {tileAt}, {c0}, {rowBytes}
            v_add_u32_e32 {o}, {tileAt}, {lastDimBytes}
            v_mov_b32_e32 {weightRead}, 0x{weightsLds:x}
            s_mov_b32 {j}, 0
            s_and_b32 {count4}, {count}, -4
            """);
        for (int r = 0; r < rows; r++)
        {
            w.L($"v_mul_f32_e32 {acc[r]}, {acc[r]}, {alpha[r]}");
        }

        w.Label(".Lt_acc");
        w.L($"s_cmp_ge_u32 {j}, {count4}");
        w.L($"s_cbranch_scc1 .Lt_acc_tail");
        var ats = new string[4];
        var vs = new string[4];
        ats[0] = o;
        for (int e = 1; e < 4; e++)
        {
            ats[e] = w.V();
            w.L($"v_add_u32_e32 {ats[e]}, {rowBytes}, {ats[e - 1]}");
        }

        for (int e = 0; e < 4; e++)
        {
            vs[e] = w.V();
            w.L($"global_load_dword {vs[e]}, {ats[e]}, {vb}");
        }

        string weights = AttentionText($"%v[{AttentionRange(w, 4)}]");
        int wFirst = AttentionFirst(weights);
        for (int r = 0; r < rows; r++)
        {
            w.L($"ds_read_b128 {weights}, {weightRead} offset:{1024 * r}");
            for (int e = 0; e < 4; e++)
            {
                w.L($"v_fma_f32 {acc[r]}, %v{wFirst + e}, {vs[e]}, {acc[r]}");
            }
        }

        w.L($"""
            v_add_u32_e32 {o}, {rowBytes}, {ats[3]}
            v_add_u32_e32 {weightRead}, 16, {weightRead}
            s_add_u32 {j}, {j}, 4
            s_branch .Lt_acc
            """);
        w.Label(".Lt_acc_tail");
        string v1 = w.V(), w1 = w.V();
        w.L($"""
            s_cmp_ge_u32 {j}, {count}
            s_cbranch_scc1 .Lt_acc_end
            global_load_dword {v1}, {o}, {vb}
            """);
        for (int r = 0; r < rows; r++)
        {
            w.L($"ds_read_b32 {w1}, {weightRead} offset:{1024 * r}");
            w.L($"v_fma_f32 {acc[r]}, {w1}, {v1}, {acc[r]}");
        }

        w.L($"""
            v_add_u32_e32 {o}, {rowBytes}, {o}
            v_add_u32_e32 {weightRead}, 4, {weightRead}
            s_add_u32 {j}, {j}, 1
            s_branch .Lt_acc_tail
            """);
        w.Label(".Lt_acc_end");
        w.L($"s_add_u32 {c0}, {c0}, 0x{AttentionTile:x}");
        w.L($"s_branch .Lt_tile");
        w.Label(".Lt_tiles_end");

        // Each row's output (the rows below rph) and its log-sum-exp, computed by every thread, stored by those of the head.
        string inverse = w.V(), output = w.V(), logSumExp = w.V(), dimSaved = w.SPair();
        for (int r = 0; r < rows; r++)
        {
            string i = w.S();
            w.L($"""
                s_add_u32 {i}, {first}, {r}
                s_cmp_ge_u32 {i}, {rph}
                s_cbranch_scc1 .Lt_out{r}
                v_rcp_f32_e32 {inverse}, {sum[r]}
                v_cmp_lt_f32_e32 vcc, 0, {sum[r]}
                v_cndmask_b32_e32 {inverse}, 0, {inverse}, vcc
                v_mul_f32_e32 {output}, {acc[r]}, {inverse}
                v_log_f32_e32 {logSumExp}, {sum[r]}
                v_fma_f32 {logSumExp}, {logSumExp}, {ln2}, {max[r]}
                v_cmp_gt_u32_e32 vcc, {dim}, {w.Tid}
                s_and_saveexec_b64 {dimSaved}, vcc
                global_store_dword {dimBytes}, {output}, {outputs[r]}
                s_or_b64 exec, exec, {dimSaved}
                s_cmp_eq_u64 {lse}, 0
                s_cbranch_scc1 .Lt_out{r}
                """);
            w.StoreFirstLane((w.ElementAddress(lse, rowIndex[r]), logSumExp));
            w.Label($".Lt_out{r}");
        }

        w.EndRows();
        return w.Source();
    }
}
