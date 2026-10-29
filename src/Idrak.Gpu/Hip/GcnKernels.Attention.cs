// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Gpu.Hip;

// HIP's attention kernels in GCN (plan 35, G4b), with CUDA's names where CUDA has the kernel (PtxKernels.Rows.cs,
// .Spans.cs, .Quantized.cs) and the CPU backend's results (CpuBackend.Quantized.cs, .CacheAttention.cs, .Spans.cs) within
// the device-against-CPU tolerance.
//
// One kernel shape serves every attention over a range of keys: decoding and the tiled form over a key/value cache
// (float32, bfloat16 or int8: attention_decode_*; the rows' range is the cache up to the device position, from the
// window's start) and attention over key ranges read from the device (attention_spans_f32: grouped heads, tables of
// ranges, and any row strides, so the strided form of a self-attention layer reads its packed projections in place).
// A block of RowThreads (four waves, declared) takes one query row and one split of its keys (grid y), walking the rows
// grid-stride (grid x). Its query row goes to LDS once; then tiles of 256 keys in order: each thread scores one key (a
// dot product over the head, the query read from LDS, four dimensions a step with four loads in flight), the block's
// maximum and sum come from RowWriter's fixed tree (rule 86), the running maximum, sum and the thread's accumulators are
// rescaled by e^(old - new) (an online softmax), the tile's weights go to LDS and each thread adds the tile's weighted
// values for its dimensions (d = tid + 256 k, k < 4: head sizes up to AttentionMaxDim) in key order, reading the values
// coalesced. So a row's result depends only on its range and the split's positions: the same bits on any grid and any
// device (no atomics).
//
// Splits (decoding's few rows over a long cache, rule 81): split s takes the positions [s · chunk, s · chunk + chunk)
// of the row's range, chunk a multiple of the tile chosen by the backend from the shapes only. With one split a block
// writes the row's output (and its log-sum-exp, the natural log of Σ e^score, when lse is not 0; -∞ and zeros for an
// empty range); with several it writes (max, sum, accumulators) to part[row, split] and attention_merge_f32 adds the
// splits in split order (a second launch instead of CUDA's last-block merge, which counts blocks with an atomic).
//
// For the profile's labels and --bench-hip's rates (HipBackend.Cost): each kernel, its arguments (pointers, then 32-bit
// values, the float ones as their bits, in this order; PackArguments packs them; a pointer of 0 for lse skips it), its
// launch, and its floating-point operations and global-memory bytes per unit of work (an exp or a reciprocal counted as
// one operation; the per-key softmax work, a few operations, left out):
//
// | Kernel                | Arguments                                                                                   | Launch (blocks of 256)            | Unit                     | Flops | Bytes |
// |-----------------------|---------------------------------------------------------------------------------------------|-----------------------------------|--------------------------|-------|-------|
// | attention_decode_f32  | q, keys, values, pos, y, lse, part; rph, steps, cap, dim, scale, window, softcap, chunk, splits, n | rows (or fewer) x splits    | row x position x dim     | 4     | 8     |
// | attention_decode_bf16 | q, keys, values, pos, y, lse, part; rph, steps, cap, dim, scale, words, window, softcap, chunk, splits, n | rows x splits         | row x position x dim     | 4     | 4     |
// | attention_decode_int8 | q, keys, values, kscales, vscales, pos, y, lse, part; rph, steps, cap, dim, scale, words, window, softcap, chunk, splits, n | rows x splits | row x position x dim | 4 | 2 |
// | attention_spans_f32   | q, keys, values, starts, ends, y, lse, part; heads, group, hpt, rows, keyrows, dim, scale, softcap, chunk, splits, qbatch, qhead, qrow, kbatch, khead, krow, ybatch, yhead, yrow, n | rows x splits | row x key x dim | 4 | 8 |
// | attention_merge_f32   | part, y, lse; splits, dim, heads, rows, ybatch, yhead, yrow, n                              | rows (or fewer)                   | row x split x dim        | 3     | 4     |
// | attn_scores_int8      | q, cache, scales, y; steps, capacity, dim, words, n                                         | one score a work-item (n)         | score x dim              | 2     | 5     |
// | attn_context_int8     | w, cache, scales, y; steps, capacity, dim, words, n                                         | one word of 4 outputs a work-item | output x position        | 2     | 3     |
//
// attention_decode_*: n = heads · rph query rows, row r of head r / rph sees positions [begin, end) with end = min(pos +
// (r mod rph) mod steps, cap - 1) + 1 and begin = end - window (0 without a window); keys and values [heads, cap, dim]
// (float32), or rows of `words` words (bfloat16 pairs, low half first; int8 bytes, four a word, with one scale per cached
// row in kscales and vscales [heads, cap]); a soft-cap (softcap > 0) maps each scaled score s to softcap · tanh(s /
// softcap). attention_spans_f32: n = batch · heads · rows, query head H = b · heads + h at row i reads its range from
// starts and ends[(H / hpt) · rows + i] (clamped to [0, keyrows]) and key/value head g = h / group; the query row is at
// q[b · qbatch + h · qhead + i · qrow], key c at keys[b · kbatch + g · khead + c · krow], the output at y[b · ybatch + h ·
// yhead + i · yrow], the log-sum-exp at lse[H · rows + i] (all in floats). attention_merge_f32 writes y the same way.
// Byte offsets within a key/value head are 32-bit (the backend checks), the others 64-bit.
internal static partial class GcnKernels
{
    /// <summary>The largest head size the attention kernels take: four dimensions a thread of the block.</summary>
    internal const int AttentionMaxDim = 1024;

    /// <summary>Keys an attention tile scores together: one a thread of the block.</summary>
    internal const int AttentionTile = RowThreads;

    /// <summary>The most splits of a row's keys (attention_merge_f32 adds them in order).</summary>
    internal const int AttentionMostSplits = 64;

    // Dimensions a thread holds: d = tid + 256 k for k below this.
    private const int AttentionSlots = AttentionMaxDim / RowThreads;

    /// <summary>
    /// The LDS an attention block declares: the reductions' two areas (2,048 bytes), a tile's weights (1,024) and the
    /// query row (4 · <see cref="AttentionMaxDim"/>). The backend runs the kernels only where a block may have this much.
    /// </summary>
    internal const int AttentionLdsBytes = 2 * 1024 + 4 * AttentionTile + 4 * AttentionMaxDim;

    /// <summary>The attention kernels, in the order they join <see cref="All"/>.</summary>
    internal static IReadOnlyList<GcnKernelSource> Attention { get; } =
    [
        AttentionRows("attention_decode_f32", AttentionCache.Float32, spans: false),
        AttentionRows("attention_decode_bf16", AttentionCache.BFloat16, spans: false),
        AttentionRows("attention_decode_int8", AttentionCache.Int8, spans: false),
        AttentionRows("attention_spans_f32", AttentionCache.Float32, spans: true),
        AttentionMerge(),
        AttentionScoresInt8(),
        AttentionContextInt8(),
    ];

    private enum AttentionCache
    {
        Float32,
        BFloat16,
        Int8,
    }

    private static string AttentionText(FormattableString text) => text.ToString(CultureInfo.InvariantCulture);

    // The reciprocal LLVM's unsigned division by a scalar count (at least 1) takes, made once, in a scalar register.
    private static string AttentionReciprocal(RowWriter w, string count)
    {
        string f = w.V(), u = w.V(), negated = w.S(), r = w.S();
        w.L($"""
            v_cvt_f32_u32_e32 {f}, {count}
            s_sub_i32 {negated}, 0, {count}
            v_rcp_iflag_f32_e32 {f}, {f}
            v_mul_f32_e32 {f}, 0x4f7ffffe, {f}
            v_cvt_u32_f32_e32 {u}, {f}
            v_readfirstlane_b32 {r}, {u}
            s_mul_i32 {negated}, {negated}, {r}
            s_mul_hi_u32 {negated}, {r}, {negated}
            s_add_i32 {r}, {r}, {negated}
            """);
        return r;
    }

    // num / count and num mod count (scalars, unsigned) through the count's reciprocal: a high product, two corrections.
    private static (string Quotient, string Remainder) AttentionDivMod(RowWriter w, string num, string count, string reciprocal)
    {
        string q = w.S(), m = w.S(), t = w.S(), next = w.S();
        w.L($"""
            s_mul_hi_u32 {q}, {num}, {reciprocal}
            s_mul_i32 {t}, {q}, {count}
            s_sub_i32 {m}, {num}, {t}
            s_add_u32 {next}, {q}, 1
            s_sub_i32 {t}, {m}, {count}
            s_cmp_ge_u32 {m}, {count}
            s_cselect_b32 {m}, {t}, {m}
            s_cselect_b32 {q}, {next}, {q}
            s_add_u32 {next}, {q}, 1
            s_sub_i32 {t}, {m}, {count}
            s_cmp_ge_u32 {m}, {count}
            s_cselect_b32 {m}, {t}, {m}
            s_cselect_b32 {q}, {next}, {q}
            """);
        return (q, m);
    }

    // a · b + c · d (+ e · f) in floats, 32-bit (an index below 2^31), in a fresh scalar register.
    private static string AttentionIndex(RowWriter w, params (string A, string B)[] terms)
    {
        string sum = w.S(), t = w.S();
        w.L($"s_mul_i32 {sum}, {terms[0].A}, {terms[0].B}");
        for (int i = 1; i < terms.Length; i++)
        {
            w.L($"s_mul_i32 {t}, {terms[i].A}, {terms[i].B}");
            w.L($"s_add_u32 {sum}, {sum}, {t}");
        }

        return sum;
    }

    // An element of a cached row as a float: bfloat16 from its word (shift 16 for the low half, 0 for the high one), int8
    // from its word's byte (at bit 0, 8, 16 or 24). Float32 values are themselves.
    private static string AttentionWiden(AttentionCache cache, string x, string shiftOrBit) => cache switch
    {
        AttentionCache.BFloat16 => AttentionText($"""
            v_lshlrev_b32_e32 {x}, {shiftOrBit}, {x}
            v_and_b32_e32 {x}, 0xffff0000, {x}
            """),
        AttentionCache.Int8 => AttentionText($"""
            v_bfe_i32 {x}, {x}, {shiftOrBit}, 8
            v_cvt_f32_i32_e32 {x}, {x}
            """),
        _ => "",
    };

    // Attention of one query row a block over its range of keys (see the file's remarks).
    private static GcnKernelSource AttentionRows(string name, AttentionCache cache, bool spans)
    {
        bool int8 = cache == AttentionCache.Int8, packed = cache != AttentionCache.Float32;
        var w = new RowWriter(name, arrays: 1, workgroupY: true, extraLds: 4 * AttentionTile + 4 * AttentionMaxDim);
        int weightsLds = w.ExtraLdsBase, queryLds = w.ExtraLdsBase + 4 * AttentionTile;
        var a = spans
            ? w.Arguments(["q", "keys", "values", "starts", "ends", "y", "lse", "part"],
                ["heads", "group", "hpt", "rows", "keyrows", "dim", "scale", "softcap", "chunk", "splits", "qbatch", "qhead", "qrow", "kbatch", "khead", "krow",
                    "ybatch", "yhead", "yrow", "n"])
            : w.Arguments(int8 ? ["q", "keys", "values", "kscales", "vscales", "pos", "y", "lse", "part"] : ["q", "keys", "values", "pos", "y", "lse", "part"],
                packed ? ["rph", "steps", "cap", "dim", "scale", "words", "window", "softcap", "chunk", "splits", "n"]
                    : ["rph", "steps", "cap", "dim", "scale", "window", "softcap", "chunk", "splits", "n"]);
        string dim = a.Scalar("dim"), scale = a.Scalar("scale"), softcap = a.Scalar("softcap"), chunk = a.Scalar("chunk"), splits = a.Scalar("splits");
        string q = a.Pointer("q"), keys = a.Pointer("keys"), values = a.Pointer("values"), y = a.Pointer("y"), lse = a.Pointer("lse"), part = a.Pointer("part");

        // Once a call (rule 85): this block's split, a cached row's bytes, the slots of the dimensions a thread holds.
        string splitFirst = w.S(), splitEnd = w.S(), rowBytes = w.S(), dimPlus2 = w.S(), ln2 = w.S(), capInverse = w.V();
        w.L($"""
            s_mul_i32 {splitFirst}, s5, {chunk}
            s_add_u32 {splitEnd}, {splitFirst}, {chunk}
            s_lshl_b32 {rowBytes}, {(spans ? a.Scalar("krow") : packed ? a.Scalar("words") : dim)}, 2
            s_add_u32 {dimPlus2}, {dim}, 2
            s_mov_b32 {ln2}, {RowWriter.Ln2}
            v_rcp_f32_e32 {capInverse}, {softcap}
            """);
        var dims = new string[AttentionSlots];
        var dimBytes = new string[AttentionSlots];
        var queryAt = new string[AttentionSlots];
        var valueAt = new string[AttentionSlots];
        var widen = new string[AttentionSlots];
        for (int k = 0; k < AttentionSlots; k++)
        {
            (dims[k], dimBytes[k], queryAt[k]) = (w.V(), w.V(), w.V());
            w.L($"v_add_u32_e32 {dims[k]}, 0x{RowThreads * k:x}, {w.Tid}");
            w.L($"v_lshlrev_b32_e32 {dimBytes[k]}, 2, {dims[k]}");
            w.L($"v_add_u32_e32 {queryAt[k]}, 0x{queryLds:x}, {dimBytes[k]}");
            string t = w.V();
            switch (cache)
            {
                case AttentionCache.BFloat16:
                    (valueAt[k], widen[k]) = (w.V(), w.V());
                    w.L($"""
                        v_lshrrev_b32_e32 {t}, 1, {dims[k]}
                        v_lshlrev_b32_e32 {valueAt[k]}, 2, {t}
                        v_and_b32_e32 {t}, 1, {dims[k]}
                        v_xor_b32_e32 {t}, 1, {t}
                        v_lshlrev_b32_e32 {widen[k]}, 4, {t}
                        """);
                    break;
                case AttentionCache.Int8:
                    (valueAt[k], widen[k]) = (w.V(), w.V());
                    w.L($"""
                        v_and_b32_e32 {valueAt[k]}, -4, {dims[k]}
                        v_and_b32_e32 {t}, 3, {dims[k]}
                        v_lshlrev_b32_e32 {widen[k]}, 3, {t}
                        """);
                    break;
                default:
                    (valueAt[k], widen[k]) = (dimBytes[k], "");
                    break;
            }
        }

        string tid4 = w.V(), weightAt = w.V();
        w.L($"v_lshlrev_b32_e32 {tid4}, 2, {w.Tid}");
        w.L($"v_add_u32_e32 {weightAt}, 0x{weightsLds:x}, {tid4}");

        // The rows' constants: for a cache, the device position and the reciprocals of rph and steps; for ranges, those of
        // rows, heads, group and hpt.
        string position = "", last = "", capRow = "", byRph = "", bySteps = "", byRows = "", byHeads = "", byGroup = "", byTables = "";
        if (spans)
        {
            (byRows, byHeads, byGroup, byTables) = (AttentionReciprocal(w, a.Scalar("rows")), AttentionReciprocal(w, a.Scalar("heads")),
                AttentionReciprocal(w, a.Scalar("group")), AttentionReciprocal(w, a.Scalar("hpt")));
        }
        else
        {
            string bits = w.S(), value = w.V();
            (position, last, capRow) = (w.S(), w.S(), w.S());
            w.L($"""
                s_load_dword {bits}, {a.Pointer("pos")}, 0x0
                v_cvt_u32_f32_e32 {value}, {bits}
                v_readfirstlane_b32 {position}, {value}
                s_sub_u32 {last}, {a.Scalar("cap")}, 1
                s_mul_i32 {capRow}, {a.Scalar("cap")}, {(packed ? a.Scalar("words") : dim)}
                """);
            (byRph, bySteps) = (AttentionReciprocal(w, a.Scalar("rph")), AttentionReciprocal(w, a.Scalar("steps")));
        }

        w.BeginRows(a.Scalar("n"));

        // This row's range [begin, end) and the offsets (in floats) of its query row, its key/value head and its output.
        string begin = w.S(), end = w.S(), queryOffset, cacheOffset, outputOffset, scaleOffset = "";
        if (spans)
        {
            var (headAll, index) = AttentionDivMod(w, w.Row, a.Scalar("rows"), byRows);
            var (batch, head) = AttentionDivMod(w, headAll, a.Scalar("heads"), byHeads);
            var (kv, _) = AttentionDivMod(w, head, a.Scalar("group"), byGroup);
            var (table, _) = AttentionDivMod(w, headAll, a.Scalar("hpt"), byTables);
            string at = AttentionIndex(w, (table, a.Scalar("rows")));
            w.L($"s_add_u32 {at}, {at}, {index}");
            string startAt = w.ElementAddress(a.Pointer("starts"), at), endAt = w.ElementAddress(a.Pointer("ends"), at);
            string first = w.V(), past = w.V(), keyRows = a.Scalar("keyrows");
            w.L($"""
                global_load_dword {first}, {w.Zero}, {startAt}
                global_load_dword {past}, {w.Zero}, {endAt}
                v_cvt_i32_f32_e32 {first}, {first}
                v_cvt_i32_f32_e32 {past}, {past}
                v_readfirstlane_b32 {begin}, {first}
                v_readfirstlane_b32 {end}, {past}
                s_max_i32 {begin}, {begin}, 0
                s_min_i32 {begin}, {begin}, {keyRows}
                s_max_i32 {end}, {end}, 0
                s_min_i32 {end}, {end}, {keyRows}
                """);
            queryOffset = AttentionIndex(w, (batch, a.Scalar("qbatch")), (head, a.Scalar("qhead")), (index, a.Scalar("qrow")));
            cacheOffset = AttentionIndex(w, (batch, a.Scalar("kbatch")), (kv, a.Scalar("khead")));
            outputOffset = AttentionIndex(w, (batch, a.Scalar("ybatch")), (head, a.Scalar("yhead")), (index, a.Scalar("yrow")));
        }
        else
        {
            var (head, index) = AttentionDivMod(w, w.Row, a.Scalar("rph"), byRph);
            var (_, step) = AttentionDivMod(w, index, a.Scalar("steps"), bySteps);
            string window = a.Scalar("window"), t = w.S();
            w.L($"""
                s_add_u32 {end}, {position}, {step}
                s_min_u32 {end}, {end}, {last}
                s_add_u32 {end}, {end}, 1
                s_sub_u32 {t}, {end}, {window}
                s_cmp_gt_u32 {end}, {window}
                s_cselect_b32 {begin}, {t}, 0
                s_cmp_eq_u32 {window}, 0
                s_cselect_b32 {begin}, 0, {begin}
                """);
            queryOffset = AttentionIndex(w, (w.Row, dim));
            outputOffset = queryOffset;
            cacheOffset = AttentionIndex(w, (head, capRow));
            if (int8)
            {
                scaleOffset = AttentionIndex(w, (head, a.Scalar("cap")));
            }
        }

        string lo = w.S(), hi = w.S();
        w.L($"s_max_u32 {lo}, {begin}, {splitFirst}");
        w.L($"s_min_u32 {hi}, {end}, {splitEnd}");
        string qb = w.ElementAddress(q, queryOffset), kb = w.ElementAddress(keys, cacheOffset), vb = w.ElementAddress(values, cacheOffset);
        string yb = w.ElementAddress(y, outputOffset);
        string ksb = int8 ? w.ElementAddress(a.Pointer("kscales"), scaleOffset) : "", vsb = int8 ? w.ElementAddress(a.Pointer("vscales"), scaleOffset) : "";

        // The query row into LDS.
        for (int k = 0; k < AttentionSlots; k++)
        {
            string saved = w.SPair(), value = w.V();
            if (k > 0)
            {
                w.L($"s_cmp_le_u32 {dim}, 0x{RowThreads * k:x}");
                w.L($"s_cbranch_scc1 .La_query{k}");
            }

            w.L($"""
                v_cmp_gt_u32_e32 vcc, {dim}, {dims[k]}
                s_and_saveexec_b64 {saved}, vcc
                global_load_dword {value}, {dimBytes[k]}, {qb}
                ds_write_b32 {queryAt[k]}, {value}
                s_or_b64 exec, exec, {saved}
                """);
            if (k > 0)
            {
                w.Label($".La_query{k}");
            }
        }

        w.L($"s_barrier");
        string max = w.V(), sum = w.V(), c0 = w.S();
        var acc = new string[AttentionSlots];
        w.L($"v_mov_b32_e32 {max}, {RowWriter.NegativeInfinity}");
        w.L($"v_mov_b32_e32 {sum}, 0");
        for (int k = 0; k < AttentionSlots; k++)
        {
            acc[k] = w.V();
            w.L($"v_mov_b32_e32 {acc[k]}, 0");
        }

        w.L($"s_mov_b32 {c0}, {lo}");
        w.Label(".La_tile");
        string count = w.S(), inTile = w.SPair(), score = w.V(), valueScale = w.V();
        w.L($"""
            s_cmp_ge_u32 {c0}, {hi}
            s_cbranch_scc1 .La_tiles_end
            s_sub_u32 {count}, {hi}, {c0}
            s_min_u32 {count}, {count}, 0x{AttentionTile:x}
            v_mov_b32_e32 {score}, {RowWriter.NegativeInfinity}
            v_mov_b32_e32 {valueScale}, 0
            v_cmp_gt_u32_e32 vcc, {count}, {w.Tid}
            s_and_saveexec_b64 {inTile}, vcc
            """);

        // Phase 1: this thread's key c0 + tid, its dot product with the query in dimension order, four a step.
        string key = w.V(), at4 = w.V(), dot = w.V(), queryRead = w.V(), d = w.S(), dim4 = w.S();
        w.L($"""
            v_add_u32_e32 {key}, {c0}, {w.Tid}
            v_mul_lo_u32 {at4}, {key}, {rowBytes}
            v_mov_b32_e32 {dot}, 0
            v_mov_b32_e32 {queryRead}, 0x{queryLds:x}
            s_mov_b32 {d}, 0
            s_and_b32 {dim4}, {dim}, -4
            """);
        w.Label(".La_dot");
        w.L($"s_cmp_ge_u32 {d}, {dim4}");
        w.L($"s_cbranch_scc1 .La_dot_end");
        string queries = AttentionText($"%v[{AttentionRange(w, 4)}]");
        int qFirst = AttentionFirst(queries);
        var xs = new string[4];
        switch (cache)
        {
            case AttentionCache.Float32:
                for (int e = 0; e < 4; e++)
                {
                    xs[e] = w.V();
                    w.L($"global_load_dword {xs[e]}, {at4}, {kb}{AttentionOffset(4 * e)}");
                }

                break;
            case AttentionCache.BFloat16:
                {
                    string w0 = w.V(), w1 = w.V();
                    w.L($"global_load_dword {w0}, {at4}, {kb}");
                    w.L($"global_load_dword {w1}, {at4}, {kb} offset:4");
                    for (int e = 0; e < 4; e++)
                    {
                        xs[e] = w.V();
                    }

                    w.L($"""
                        v_lshlrev_b32_e32 {xs[0]}, 16, {w0}
                        v_and_b32_e32 {xs[1]}, 0xffff0000, {w0}
                        v_lshlrev_b32_e32 {xs[2]}, 16, {w1}
                        v_and_b32_e32 {xs[3]}, 0xffff0000, {w1}
                        """);
                    break;
                }

            default:
                {
                    string w0 = w.V();
                    w.L($"global_load_dword {w0}, {at4}, {kb}");
                    for (int e = 0; e < 4; e++)
                    {
                        xs[e] = w.V();
                        w.L($"v_bfe_i32 {xs[e]}, {w0}, {8 * e}, 8");
                        w.L($"v_cvt_f32_i32_e32 {xs[e]}, {xs[e]}");
                    }

                    break;
                }
        }

        w.L($"ds_read_b128 {queries}, {queryRead}");
        for (int e = 0; e < 4; e++)
        {
            w.L($"v_fma_f32 {dot}, %v{qFirst + e}, {xs[e]}, {dot}");
        }

        int keyStep = cache switch { AttentionCache.Float32 => 16, AttentionCache.BFloat16 => 8, _ => 4 };
        w.L($"""
            v_add_u32_e32 {at4}, {keyStep}, {at4}
            v_add_u32_e32 {queryRead}, 16, {queryRead}
            s_add_u32 {d}, {d}, 4
            s_branch .La_dot
            """);
        w.Label(".La_dot_end");
        string rest = w.S();
        w.L($"s_and_b32 {rest}, {dim}, 3");
        for (int e = 0; e < 3; e++)
        {
            string x = w.V(), qe = w.V();
            w.L($"s_cmp_le_u32 {rest}, {e}");
            w.L($"s_cbranch_scc1 .La_dot_tail");
            switch (cache)
            {
                case AttentionCache.Float32:
                    w.L($"global_load_dword {x}, {at4}, {kb}{AttentionOffset(4 * e)}");
                    break;
                case AttentionCache.BFloat16:
                    w.L($"global_load_dword {x}, {at4}, {kb}{AttentionOffset(4 * (e >> 1))}");
                    w.L($"{(e % 2 == 0 ? AttentionText($"v_lshlrev_b32_e32 {x}, 16, {x}") : AttentionText($"v_and_b32_e32 {x}, 0xffff0000, {x}"))}");
                    break;
                default:
                    w.L($"global_load_dword {x}, {at4}, {kb}");
                    w.L($"v_bfe_i32 {x}, {x}, {8 * e}, 8");
                    w.L($"v_cvt_f32_i32_e32 {x}, {x}");
                    break;
            }

            w.L($"ds_read_b32 {qe}, {queryRead}{AttentionOffset(4 * e)}");
            w.L($"v_fma_f32 {dot}, {qe}, {x}, {dot}");
        }

        w.Label(".La_dot_tail");
        if (int8)
        {
            string keyScale = w.V(), at = w.V();
            w.L($"""
                v_lshlrev_b32_e32 {at}, 2, {key}
                global_load_dword {keyScale}, {at}, {ksb}
                global_load_dword {valueScale}, {at}, {vsb}
                v_mul_f32_e32 {dot}, {dot}, {keyScale}
                """);
        }

        string capped = w.V();
        w.L($"""
            v_mul_f32_e32 {score}, {scale}, {dot}
            s_cmp_gt_i32 {softcap}, 0
            s_cbranch_scc0 .La_capped
            v_mul_f32_e32 {capped}, {score}, {capInverse}
            v_mul_f32_e32 {capped}, 0x4038aa3b, {capped}
            v_exp_f32_e32 {capped}, {capped}
            v_add_f32_e32 {capped}, 1.0, {capped}
            v_rcp_f32_e32 {capped}, {capped}
            v_fma_f32 {capped}, {capped}, -2.0, 1.0
            v_mul_f32_e32 {score}, {softcap}, {capped}
            """);
        w.Label(".La_capped");
        w.L($"s_or_b64 exec, exec, {inTile}");

        // Phase 2: the tile's maximum, the rescale of what came before, the weights (to LDS) and their sum.
        string best = w.V(), top = w.V(), shift = w.V(), alpha = w.V(), p = w.V(), total = w.V(), tileAt = w.S();
        w.L($"v_mov_b32_e32 {best}, {score}");
        w.Reduce(RowWriter.Max, best);
        w.L($"""
            v_max_f32_e32 {top}, {max}, {best}
            v_cmp_eq_f32_e32 vcc, {RowWriter.NegativeInfinity}, {top}
            v_cndmask_b32_e64 {shift}, {top}, 0, vcc
            v_sub_f32_e32 {alpha}, {max}, {shift}
            v_mul_f32_e32 {alpha}, {RowWriter.Log2E}, {alpha}
            v_exp_f32_e32 {alpha}, {alpha}
            {RowWriter.Exp(p, score, shift)}
            """);
        if (int8)
        {
            string weighted = w.V();
            w.L($"v_mul_f32_e32 {weighted}, {p}, {valueScale}");
            w.L($"ds_write_b32 {weightAt}, {weighted}");
        }
        else
        {
            w.L($"ds_write_b32 {weightAt}, {p}");
        }

        w.L($"v_mov_b32_e32 {total}, {p}");
        w.Reduce(RowWriter.Add, total);
        w.L($"""
            v_fma_f32 {sum}, {sum}, {alpha}, {total}
            v_mov_b32_e32 {max}, {top}
            s_mul_i32 {tileAt}, {c0}, {rowBytes}
            """);

        // Phase 3: each thread's dimensions rescaled, then the tile's weighted values added in key order, four a step.
        for (int k = 0; k < AttentionSlots; k++)
        {
            string saved = w.SPair(), o = w.V(), weightRead = w.V(), j = w.S(), count4 = w.S();
            if (k > 0)
            {
                w.L($"s_cmp_le_u32 {dim}, 0x{RowThreads * k:x}");
                w.L($"s_cbranch_scc1 .La_slot{k}");
            }

            w.L($"""
                v_cmp_gt_u32_e32 vcc, {dim}, {dims[k]}
                s_and_saveexec_b64 {saved}, vcc
                v_mul_f32_e32 {acc[k]}, {acc[k]}, {alpha}
                v_add_u32_e32 {o}, {tileAt}, {valueAt[k]}
                v_mov_b32_e32 {weightRead}, 0x{weightsLds:x}
                s_mov_b32 {j}, 0
                s_and_b32 {count4}, {count}, -4
                """);
            w.Label($".La_acc{k}");
            w.L($"s_cmp_ge_u32 {j}, {count4}");
            w.L($"s_cbranch_scc1 .La_acc{k}_tail");
            var at = new string[4];
            var vs = new string[4];
            at[0] = o;
            for (int e = 1; e < 4; e++)
            {
                at[e] = w.V();
                w.L($"v_add_u32_e32 {at[e]}, {rowBytes}, {at[e - 1]}");
            }

            for (int e = 0; e < 4; e++)
            {
                vs[e] = w.V();
                w.L($"global_load_dword {vs[e]}, {at[e]}, {vb}");
            }

            string weights = AttentionText($"%v[{AttentionRange(w, 4)}]");
            int wFirst = AttentionFirst(weights);
            w.L($"ds_read_b128 {weights}, {weightRead}");
            for (int e = 0; e < 4; e++)
            {
                w.L($"{AttentionWiden(cache, vs[e], widen[k])}");
                w.L($"v_fma_f32 {acc[k]}, %v{wFirst + e}, {vs[e]}, {acc[k]}");
            }

            w.L($"""
                v_add_u32_e32 {o}, {rowBytes}, {at[3]}
                v_add_u32_e32 {weightRead}, 16, {weightRead}
                s_add_u32 {j}, {j}, 4
                s_branch .La_acc{k}
                """);
            w.Label($".La_acc{k}_tail");
            string v1 = w.V(), w1 = w.V();
            w.L($"""
                s_cmp_ge_u32 {j}, {count}
                s_cbranch_scc1 .La_acc{k}_end
                global_load_dword {v1}, {o}, {vb}
                ds_read_b32 {w1}, {weightRead}
                {AttentionWiden(cache, v1, widen[k])}
                v_fma_f32 {acc[k]}, {w1}, {v1}, {acc[k]}
                v_add_u32_e32 {o}, {rowBytes}, {o}
                v_add_u32_e32 {weightRead}, 4, {weightRead}
                s_add_u32 {j}, {j}, 1
                s_branch .La_acc{k}_tail
                """);
            w.Label($".La_acc{k}_end");
            w.L($"s_or_b64 exec, exec, {saved}");
            if (k > 0)
            {
                w.Label($".La_slot{k}");
            }
        }

        w.L($"s_add_u32 {c0}, {c0}, 0x{AttentionTile:x}");
        w.L($"s_branch .La_tile");
        w.Label(".La_tiles_end");

        // The row's output (one split), or this split's part.
        string inverse = w.V();
        w.L($"""
            s_cmp_eq_u32 {splits}, 1
            s_cbranch_scc0 .La_part
            v_rcp_f32_e32 {inverse}, {sum}
            v_cmp_lt_f32_e32 vcc, 0, {sum}
            v_cndmask_b32_e32 {inverse}, 0, {inverse}, vcc
            """);
        for (int k = 0; k < AttentionSlots; k++)
        {
            string saved = w.SPair(), o = w.V();
            if (k > 0)
            {
                w.L($"s_cmp_le_u32 {dim}, 0x{RowThreads * k:x}");
                w.L($"s_cbranch_scc1 .La_out{k}");
            }

            w.L($"""
                v_cmp_gt_u32_e32 vcc, {dim}, {dims[k]}
                s_and_saveexec_b64 {saved}, vcc
                v_mul_f32_e32 {o}, {acc[k]}, {inverse}
                global_store_dword {dimBytes[k]}, {o}, {yb}
                s_or_b64 exec, exec, {saved}
                """);
            if (k > 0)
            {
                w.Label($".La_out{k}");
            }
        }

        string logSumExp = w.V();
        w.L($"""
            s_cmp_eq_u64 {lse}, 0
            s_cbranch_scc1 .La_row_end
            v_log_f32_e32 {logSumExp}, {sum}
            v_fma_f32 {logSumExp}, {logSumExp}, {ln2}, {max}
            """);
        w.StoreFirstLane((w.ElementAddress(lse, w.Row), logSumExp));
        w.L($"s_branch .La_row_end");
        w.Label(".La_part");
        string partIndex = w.S();
        w.L($"""
            s_mul_i32 {partIndex}, {w.Row}, {splits}
            s_add_u32 {partIndex}, {partIndex}, s5
            s_mul_i32 {partIndex}, {partIndex}, {dimPlus2}
            """);
        string pb = w.ElementAddress(part, partIndex), first0 = w.SPair();
        w.L($"""
            v_cmp_eq_u32_e32 vcc, 0, {w.Tid}
            s_and_saveexec_b64 {first0}, vcc
            global_store_dword {w.Zero}, {max}, {pb}
            global_store_dword {w.Zero}, {sum}, {pb} offset:4
            s_or_b64 exec, exec, {first0}
            """);
        for (int k = 0; k < AttentionSlots; k++)
        {
            string saved = w.SPair();
            if (k > 0)
            {
                w.L($"s_cmp_le_u32 {dim}, 0x{RowThreads * k:x}");
                w.L($"s_cbranch_scc1 .La_partout{k}");
            }

            w.L($"""
                v_cmp_gt_u32_e32 vcc, {dim}, {dims[k]}
                s_and_saveexec_b64 {saved}, vcc
                global_store_dword {dimBytes[k]}, {acc[k]}, {pb} offset:8
                s_or_b64 exec, exec, {saved}
                """);
            if (k > 0)
            {
                w.Label($".La_partout{k}");
            }
        }

        w.Label(".La_row_end");
        w.EndRows();
        return w.Source();
    }

    // A range of adjacent fresh vector registers ("first:last") from the writer, and its first register's number.
    private static string AttentionRange(RowWriter w, int count)
    {
        string first = w.V();
        for (int i = 1; i < count; i++)
        {
            w.V();
        }

        int id = AttentionFirst(first);
        return AttentionText($"{id}:{id + count - 1}");
    }

    private static int AttentionFirst(string register)
    {
        int at = 0;
        while (!char.IsAsciiDigit(register[at]))
        {
            at++;
        }

        int stop = at;
        while (stop < register.Length && char.IsAsciiDigit(register[stop]))
        {
            stop++;
        }

        return int.Parse(register.AsSpan(at, stop - at), CultureInfo.InvariantCulture);
    }

    private static string AttentionOffset(int bytes) => bytes == 0 ? "" : AttentionText($" offset:{bytes}");

    // The splits of rows (part[row, split] = max, sum, accumulators) merged in split order: M = the largest max, each
    // split's terms scaled by e^(max - M), y = Σ accumulators / Σ sums (zeros and a log-sum-exp of -∞ for an empty row).
    private static GcnKernelSource AttentionMerge()
    {
        var w = new RowWriter("attention_merge_f32", arrays: 0);
        var a = w.Arguments(["part", "y", "lse"], ["splits", "dim", "heads", "rows", "ybatch", "yhead", "yrow", "n"]);
        string part = a.Pointer("part"), y = a.Pointer("y"), lse = a.Pointer("lse");
        string splits = a.Scalar("splits"), dim = a.Scalar("dim");
        string dimPlus2 = w.S(), stride = w.S(), ln2 = w.S();
        w.L($"""
            s_add_u32 {dimPlus2}, {dim}, 2
            s_lshl_b32 {stride}, {dimPlus2}, 2
            s_mov_b32 {ln2}, {RowWriter.Ln2}
            """);
        string byRows = AttentionReciprocal(w, a.Scalar("rows")), byHeads = AttentionReciprocal(w, a.Scalar("heads"));
        w.BeginRows(a.Scalar("n"));
        string partIndex = AttentionIndex(w, (w.Row, splits));
        w.L($"s_mul_i32 {partIndex}, {partIndex}, {dimPlus2}");
        string pb = w.ElementAddress(part, partIndex);
        string max = w.V(), offset = w.V(), m = w.V(), s = w.S();
        w.L($"""
            v_mov_b32_e32 {max}, {RowWriter.NegativeInfinity}
            v_mov_b32_e32 {offset}, 0
            s_mov_b32 {s}, 0
            """);
        w.Label(".Lm_max");
        w.L($"""
            s_cmp_ge_u32 {s}, {splits}
            s_cbranch_scc1 .Lm_max_end
            global_load_dword {m}, {offset}, {pb}
            v_max_f32_e32 {max}, {max}, {m}
            v_add_u32_e32 {offset}, {stride}, {offset}
            s_add_u32 {s}, {s}, 1
            s_branch .Lm_max
            """);
        w.Label(".Lm_max_end");
        string shift = w.V(), sum = w.V(), l = w.V(), e = w.V();
        w.L($"""
            v_cmp_eq_f32_e32 vcc, {RowWriter.NegativeInfinity}, {max}
            v_cndmask_b32_e64 {shift}, {max}, 0, vcc
            v_mov_b32_e32 {sum}, 0
            v_mov_b32_e32 {offset}, 0
            s_mov_b32 {s}, 0
            """);
        w.Label(".Lm_sum");
        w.L($"""
            s_cmp_ge_u32 {s}, {splits}
            s_cbranch_scc1 .Lm_sum_end
            global_load_dword {m}, {offset}, {pb}
            global_load_dword {l}, {offset}, {pb} offset:4
            {RowWriter.Exp(e, m, shift)}
            v_fma_f32 {sum}, {l}, {e}, {sum}
            v_add_u32_e32 {offset}, {stride}, {offset}
            s_add_u32 {s}, {s}, 1
            s_branch .Lm_sum
            """);
        w.Label(".Lm_sum_end");
        string inverse = w.V();
        w.L($"""
            v_rcp_f32_e32 {inverse}, {sum}
            v_cmp_lt_f32_e32 vcc, 0, {sum}
            v_cndmask_b32_e32 {inverse}, 0, {inverse}, vcc
            """);
        var (headAll, index) = AttentionDivMod(w, w.Row, a.Scalar("rows"), byRows);
        var (batch, head) = AttentionDivMod(w, headAll, a.Scalar("heads"), byHeads);
        string outputOffset = AttentionIndex(w, (batch, a.Scalar("ybatch")), (head, a.Scalar("yhead")), (index, a.Scalar("yrow")));
        string yb = w.ElementAddress(y, outputOffset), logSumExp = w.V();
        w.L($"""
            s_cmp_eq_u64 {lse}, 0
            s_cbranch_scc1 .Lm_lse_done
            v_log_f32_e32 {logSumExp}, {sum}
            v_fma_f32 {logSumExp}, {logSumExp}, {ln2}, {max}
            """);
        w.StoreFirstLane((w.ElementAddress(lse, w.Row), logSumExp));
        w.Label(".Lm_lse_done");
        string d0 = w.S(), dd = w.V(), db = w.V(), acc = w.V(), o = w.V(), x = w.V(), saved = w.SPair();
        w.L($"s_mov_b32 {d0}, 0");
        w.Label(".Lm_dims");
        w.L($"""
            s_cmp_ge_u32 {d0}, {dim}
            s_cbranch_scc1 .Lm_dims_end
            v_add_u32_e32 {dd}, {d0}, {w.Tid}
            v_cmp_gt_u32_e32 vcc, {dim}, {dd}
            s_and_saveexec_b64 {saved}, vcc
            v_lshlrev_b32_e32 {db}, 2, {dd}
            v_mov_b32_e32 {acc}, 0
            v_mov_b32_e32 {o}, {db}
            v_mov_b32_e32 {offset}, 0
            s_mov_b32 {s}, 0
            """);
        w.Label(".Lm_acc");
        w.L($"""
            s_cmp_ge_u32 {s}, {splits}
            s_cbranch_scc1 .Lm_acc_end
            global_load_dword {m}, {offset}, {pb}
            global_load_dword {x}, {o}, {pb} offset:8
            {RowWriter.Exp(e, m, shift)}
            v_fma_f32 {acc}, {x}, {e}, {acc}
            v_add_u32_e32 {o}, {stride}, {o}
            v_add_u32_e32 {offset}, {stride}, {offset}
            s_add_u32 {s}, {s}, 1
            s_branch .Lm_acc
            """);
        w.Label(".Lm_acc_end");
        w.L($"""
            v_mul_f32_e32 {acc}, {acc}, {inverse}
            global_store_dword {db}, {acc}, {yb}
            s_or_b64 exec, exec, {saved}
            s_add_u32 {d0}, {d0}, 0x{RowThreads:x}
            s_branch .Lm_dims
            """);
        w.Label(".Lm_dims_end");
        w.EndRows();
        return w.Source();
    }

    // scores[r, t, c] = scales[r, c] · Σ_d q[r, t, d] · keys[r, c, d] over an int8 cache: one score a work-item.
    private static GcnKernelSource AttentionScoresInt8()
    {
        var w = new ElementWriter("attn_scores_int8", ["q", "cache", "scales", "y"], ["steps", "capacity", "dim", "words", "n"]);
        string steps = w.Scalar("steps"), capacity = w.Scalar("capacity"), dim = w.Scalar("dim");
        string byCapacity = w.Reciprocal(capacity), bySteps = w.Reciprocal(steps), wordBytes = w.S(), rowBytes = w.S();
        w.L($"s_lshl_b32 {wordBytes}, {w.Scalar("words")}, 2");
        w.L($"s_lshl_b32 {rowBytes}, {dim}, 2");
        w.Begin();
        var (rowStep, position) = w.DivMod(w.Item, capacity, byCapacity);
        var (row, _) = w.DivMod(rowStep, steps, bySteps);
        string slot = w.V(), keyAt = w.V(), queryAt = w.V(), dot = w.V(), word = w.V(), d = w.S(), lane = w.S();
        w.L($"""
            v_mul_lo_u32 {slot}, {row}, {capacity}
            v_add_u32_e32 {slot}, {slot}, {position}
            v_mul_lo_u32 {keyAt}, {slot}, {wordBytes}
            v_mul_lo_u32 {queryAt}, {rowStep}, {rowBytes}
            v_mov_b32_e32 {dot}, 0
            s_mov_b32 {d}, 0
            .Lwords:
            s_cmp_ge_u32 {d}, {dim}
            s_cbranch_scc1 .Lwords_end
            global_load_dword {word}, {keyAt}, {w.Pointer("cache")}
            """);
        for (int e = 0; e < 4; e++)
        {
            string qe = w.V(), b = w.V();
            w.L($"""
                s_add_u32 {lane}, {d}, {e}
                s_cmp_ge_u32 {lane}, {dim}
                s_cbranch_scc1 .Lnext
                global_load_dword {qe}, {queryAt}, {w.Pointer("q")}{AttentionOffset(4 * e)}
                v_bfe_i32 {b}, {word}, {8 * e}, 8
                v_cvt_f32_i32_e32 {b}, {b}
                v_fma_f32 {dot}, {qe}, {b}, {dot}
                """);
        }

        string scaleAt = w.V(), scaleValue = w.V(), at = w.V();
        w.L($"""
            .Lnext:
            v_add_u32_e32 {keyAt}, 4, {keyAt}
            v_add_u32_e32 {queryAt}, 16, {queryAt}
            s_add_u32 {d}, {d}, 4
            s_branch .Lwords
            .Lwords_end:
            v_lshlrev_b32_e32 {scaleAt}, 2, {slot}
            global_load_dword {scaleValue}, {scaleAt}, {w.Pointer("scales")}
            v_mul_f32_e32 {dot}, {dot}, {scaleValue}
            v_lshlrev_b32_e32 {at}, 2, {w.Item}
            global_store_dword {at}, {dot}, {w.Pointer("y")}
            """);
        return w.End();
    }

    // y[r, t, 4 wi .. 4 wi + 3] = Σ_c weights[r, t, c] · scales[r, c] · values[r, c, 4 wi ..] over an int8 cache, in position
    // order: one word of four outputs a work-item.
    private static GcnKernelSource AttentionContextInt8()
    {
        var w = new ElementWriter("attn_context_int8", ["w", "cache", "scales", "y"], ["steps", "capacity", "dim", "words", "n"]);
        string steps = w.Scalar("steps"), capacity = w.Scalar("capacity"), dim = w.Scalar("dim"), words = w.Scalar("words");
        string byWords = w.Reciprocal(words), bySteps = w.Reciprocal(steps), wordBytes = w.S();
        w.L($"s_lshl_b32 {wordBytes}, {words}, 2");
        w.Begin();
        var (rowStep, wordIndex) = w.DivMod(w.Item, words, byWords);
        var (row, _) = w.DivMod(rowStep, steps, bySteps);
        string weightAt = w.V(), scaleAt = w.V(), cacheAt = w.V(), t = w.V(), c = w.S();
        var acc = new string[4];
        w.L($"""
            v_mul_lo_u32 {weightAt}, {rowStep}, {capacity}
            v_lshlrev_b32_e32 {weightAt}, 2, {weightAt}
            v_mul_lo_u32 {scaleAt}, {row}, {capacity}
            v_mul_lo_u32 {cacheAt}, {scaleAt}, {wordBytes}
            v_lshlrev_b32_e32 {scaleAt}, 2, {scaleAt}
            v_lshlrev_b32_e32 {t}, 2, {wordIndex}
            v_add_u32_e32 {cacheAt}, {cacheAt}, {t}
            """);
        for (int e = 0; e < 4; e++)
        {
            acc[e] = w.V();
            w.L($"v_mov_b32_e32 {acc[e]}, 0");
        }

        string weight = w.V(), scaleValue = w.V(), word = w.V(), factor = w.V();
        w.L($"""
            s_mov_b32 {c}, 0
            .Lpositions:
            s_cmp_ge_u32 {c}, {capacity}
            s_cbranch_scc1 .Lpositions_end
            global_load_dword {weight}, {weightAt}, {w.Pointer("w")}
            global_load_dword {scaleValue}, {scaleAt}, {w.Pointer("scales")}
            global_load_dword {word}, {cacheAt}, {w.Pointer("cache")}
            v_mul_f32_e32 {factor}, {weight}, {scaleValue}
            """);
        for (int e = 0; e < 4; e++)
        {
            string b = w.V();
            w.L($"""
                v_bfe_i32 {b}, {word}, {8 * e}, 8
                v_cvt_f32_i32_e32 {b}, {b}
                v_fma_f32 {acc[e]}, {factor}, {b}, {acc[e]}
                """);
        }

        string first = w.V(), at = w.V();
        w.L($"""
            v_add_u32_e32 {weightAt}, 4, {weightAt}
            v_add_u32_e32 {scaleAt}, 4, {scaleAt}
            v_add_u32_e32 {cacheAt}, {wordBytes}, {cacheAt}
            s_add_u32 {c}, {c}, 1
            s_branch .Lpositions
            .Lpositions_end:
            v_mul_lo_u32 {first}, {rowStep}, {dim}
            v_lshlrev_b32_e32 {t}, 2, {wordIndex}
            v_add_u32_e32 {first}, {first}, {t}
            v_lshlrev_b32_e32 {at}, 2, {first}
            """);
        for (int e = 0; e < 4; e++)
        {
            string lane = w.V(), saved = w.SPair();
            w.L($"""
                v_add_u32_e32 {lane}, {e}, {t}
                v_cmp_gt_u32_e32 vcc, {dim}, {lane}
                s_and_saveexec_b64 {saved}, vcc
                global_store_dword {at}, {acc[e]}, {w.Pointer("y")}{AttentionOffset(4 * e)}
                s_or_b64 exec, exec, {saved}
                """);
        }

        return w.End();
    }
}
