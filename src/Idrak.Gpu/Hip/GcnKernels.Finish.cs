// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Gpu.Hip;

// The last block's finish (plan 35, G6b; rule 82: one launch where the splits of a sum, or a fused kernel's tail, took a
// second): a block that has stored its part bumps a completion counter, and the block that finds itself last does the
// rest (adds the parts in split order, normalizes the rows, writes the pair's activation) in the same launch, as CUDA's
// kernels do with atom.global.add. Rule 86 forbids adding values with atomics, not signalling with them: the counter only
// says which block is last; every sum is still taken in the order the sizes fix, by one block, so a result is the same on
// every grid, every order the blocks finish in and every device. The counter resets itself (the last block stores 0), so
// the next launch, and a replay of the same launch, finds it at zero; HipBackend keeps one zeroed buffer of counters per
// device from its pool (rule 77) and gives each launch its own range.
//
// The memory order is gfx9's (LLVM's AMDGPU memory model, agent scope): a block's stores complete (s_waitcnt vmcnt(0),
// written out: no register waits for a store) before the atomic, which returns the count (glc) to lane 0; the last block
// reads what the others stored with glc loads, which miss the compute unit's vector L1 and read the L2 every compute
// unit shares (CUDA's ld.global.cg). The interpreter checks each step: a load of another block's store without the
// atomic between them, or without glc, is reported as a race.
internal static partial class GcnKernels
{
    private static string FinishText(FormattableString text) => text.ToString(CultureInfo.InvariantCulture);

    /// <summary>
    /// A one-wave block's arrival at counters[<paramref name="index"/>] (a scalar register): its stores completed, lane 0
    /// adds 1 and reads the count before (glc). A block that is not the <paramref name="total"/>-th (a scalar register or a
    /// small constant) branches to <paramref name="notLast"/>; the last resets the counter and goes on.
    /// </summary>
    private static void WaveArrive(ProductWriter w, string counters, string index, string total, string notLast)
    {
        string offset = w.S(), at = w.V(), one = w.V(), old = w.V(), count = w.S(), saved = w.SPair(), zero = w.V();
        w.L($"s_waitcnt vmcnt(0)                                       ; this block's stores done: the atomic releases them");
        w.L($"s_lshl_b32 {offset}, {index}, 2");
        w.L($"v_mov_b32_e32 {at}, {offset}");
        // Lane 0 alone through saveexec and back (EXEC known full after: the allocator keeps the registers free).
        w.L($"v_mov_b32_e32 {one}, 1");
        w.L($"v_mov_b32_e32 {old}, 0                                    ; defined under the whole EXEC (see WaveColumns)");
        w.L($"v_mov_b32_e32 {zero}, 0");
        w.L($"v_cmp_eq_u32_e32 vcc, 0, {w.Tid}");
        w.L($"s_and_saveexec_b64 {saved}, vcc");
        w.L($"global_atomic_add {old}, {at}, {one}, {counters} glc        ; lane 0: the blocks that arrived before");
        w.L($"s_or_b64 exec, exec, {saved}");
        w.L($"v_readfirstlane_b32 {count}, {old}");
        w.L($"s_add_u32 {count}, {count}, 1");
        w.L($"s_cmp_lg_u32 {count}, {total}");
        w.L($"s_cbranch_scc1 {notLast}");
        w.L($"s_and_saveexec_b64 {saved}, vcc");
        w.L($"global_store_dword {at}, {zero}, {counters}               ; the last block resets the counter");
        w.L($"s_or_b64 exec, exec, {saved}");
    }

    /// <summary>
    /// A block's arrival (every wave of a <see cref="RowThreads"/> block) at counters[<paramref name="index"/>]: each wave's
    /// stores completed and the waves met at a barrier, thread 0 adds 1 (glc) and leaves the count after in LDS at
    /// <paramref name="slot"/> for every wave; a block that is not the <paramref name="total"/>-th branches to
    /// <paramref name="notLast"/>, the last resets the counter and goes on.
    /// </summary>
    private static void BlockArrive(RowWriter w, string counters, string index, string total, string notLast, int slot)
    {
        string offset = w.S(), at = w.V(), one = w.V(), old = w.V(), seen = w.V(), count = w.S(), saved = w.SPair();
        w.L($"""
            s_waitcnt vmcnt(0)
            s_barrier
            s_lshl_b32 {offset}, {index}, 2
            v_mov_b32_e32 {at}, {offset}
            v_mov_b32_e32 {one}, 1
            v_mov_b32_e32 {old}, 0
            v_cmp_eq_u32_e32 vcc, 0, {w.Tid}
            s_and_saveexec_b64 {saved}, vcc
            global_atomic_add {old}, {at}, {one}, {counters} glc
            v_add_u32_e32 {old}, 1, {old}
            ds_write_b32 {w.Zero}, {old} offset:{slot}
            s_or_b64 exec, exec, {saved}
            s_barrier
            ds_read_b32 {seen}, {w.Zero} offset:{slot}
            v_readfirstlane_b32 {count}, {seen}
            s_cmp_lg_u32 {count}, {total}
            s_cbranch_scc1 {notLast}
            v_cmp_eq_u32_e32 vcc, 0, {w.Tid}
            s_and_saveexec_b64 {saved}, vcc
            global_store_dword {at}, {w.Zero}, {counters}
            s_or_b64 exec, exec, {saved}
            """);
    }

    // ---- The last block's add and norms on one wave ----------------------------------------------------------------

    // Threads of the row kernel a wave stands in for: lane l takes the threads l, l + 64, l + 128 and l + 192.
    private const int WaveThreads = RowThreads / 64;

    /// <summary>
    /// A column loop of a <see cref="RowThreads"/> row kernel (RowWriter.Columns) on one wave: lane l takes the threads
    /// l + 64 j (j &lt; 4), each thread its columns t, t + 256, … in order, so each thread's partial is summed exactly as
    /// the row kernel's; <paramref name="blocks"/> blocks of 256 columns a step (their loads first, every one in flight),
    /// then one block at a time with each thread's lanes past <paramref name="cols"/> left out. Each wave instruction
    /// reads 64 consecutive floats (coalesced). The body's element: (i · 4 + j) indexes its registers, the vector byte
    /// offset of the step and the instruction offset of (i, j).
    /// </summary>
    private static void WaveColumns(ProductWriter w, string cols, string laneBytes, int blocks, Func<int, string, string, string> load,
        Func<int, string, string, string> use, IEnumerable<string> written)
    {
        // The registers the body writes, defined first under the whole EXEC: the tail writes them under a narrowed one,
        // and a register so written is live from its last definition (here, not the kernel's start: the product's
        // registers stay free, so its occupancy does).
        foreach (string register in written)
        {
            w.L($"v_mov_b32_e32 {register}, 0");
        }

        string label = w.NewLabel(), b = w.S(), end = w.S(), b4 = w.S(), off = w.V();
        w.L($"s_mov_b32 {b}, 0");
        w.Label(label + "_main");
        w.L($"s_add_u32 {end}, {b}, 0x{RowThreads * blocks:x}");
        w.L($"s_cmp_gt_u32 {end}, {cols}");
        w.L($"s_cbranch_scc1 {label}_tail");
        w.L($"s_lshl_b32 {b4}, {b}, 2");
        w.L($"v_add_u32_e32 {off}, {b4}, {laneBytes}");
        for (int i = 0; i < blocks; i++)
        {
            for (int j = 0; j < WaveThreads; j++)
            {
                w.Block(load(i * WaveThreads + j, off, ProductWriter.Offset(4 * (RowThreads * i + 64 * j))));
            }
        }

        for (int i = 0; i < blocks; i++)
        {
            for (int j = 0; j < WaveThreads; j++)
            {
                w.Block(use(i * WaveThreads + j, off, ProductWriter.Offset(4 * (RowThreads * i + 64 * j))));
            }
        }

        w.L($"s_mov_b32 {b}, {end}");
        w.L($"s_branch {label}_main");
        w.Label(label + "_tail");
        w.L($"s_cmp_ge_u32 {b}, {cols}");
        w.L($"s_cbranch_scc1 {label}_end");
        w.L($"s_lshl_b32 {b4}, {b}, 2");
        w.L($"v_add_u32_e32 {off}, {b4}, {laneBytes}");
        string first = w.V(), column = w.V(), saved = w.SPair();
        w.L($"v_add_u32_e32 {first}, {b}, {w.Tid}");
        for (int j = 0; j < WaveThreads; j++)
        {
            w.L($"v_add_u32_e32 {column}, {64 * j}, {first}");
            w.L($"v_cmp_gt_u32_e32 vcc, {cols}, {column}");
            w.L($"s_and_saveexec_b64 {saved}, vcc");
            w.Block(load(j, off, ProductWriter.Offset(4 * 64 * j)));
            w.Block(use(j, off, ProductWriter.Offset(4 * 64 * j)));
            w.L($"s_or_b64 exec, exec, {saved}");
        }

        w.L($"s_add_u32 {b}, {b}, 0x{RowThreads:x}");
        w.L($"s_branch {label}_tail");
        w.Label(label + "_end");
    }

    /// <summary>
    /// RowWriter.Reduce(Add) of a <see cref="RowThreads"/> block on one wave, its sums in the same tree: threads below 128
    /// add the one 128 above, below 64 the one 64 above (both within a lane: <paramref name="held"/>[0] += [2], [1] += [3],
    /// then [0] += [1]), then 32, 16, …, 1 above across the lanes (ds_bpermute); the result, lane 0's, into a scalar register.
    /// </summary>
    private static string WaveReduceAdd(ProductWriter w, string[] held, string laneBytes)
    {
        string other = w.V(), source = w.V(), result = w.S();
        w.L($"v_add_f32_e32 {held[0]}, {held[0]}, {held[2]}");
        w.L($"v_add_f32_e32 {held[1]}, {held[1]}, {held[3]}");
        w.L($"v_add_f32_e32 {held[0]}, {held[0]}, {held[1]}");
        for (int stride = 32; stride >= 1; stride /= 2)
        {
            w.L($"v_add_u32_e32 {source}, {4 * stride}, {laneBytes}");
            w.L($"ds_bpermute_b32 {other}, {source}, {held[0]}");
            w.L($"v_add_f32_e32 {held[0]}, {held[0]}, {other}");
        }

        w.L($"v_readfirstlane_b32 {result}, {held[0]}");
        return result;
    }

    /// <summary>
    /// The packed product's add and norms (Backend.PackedMatMulAddRmsNormKernel and its sandwich form), on the last block
    /// once every column of y is final: for each row, y's own RMS norm · (ygain + yoffset) first when flags bit 0 is set,
    /// sum = residual + y, normalized = RmsNormAffine(sum, gain, eps, offset); y, sum and normalized written. The passes,
    /// reductions and operations are gemv_addnorm_f32's for one row (a block of 256 there), so the three outputs equal
    /// the product followed by gemv_addnorm_f32 bit for bit (y read back with glc: other blocks wrote it).
    /// </summary>
    private static void AddNormTail(ProductWriter w, string y, string m, string n)
    {
        string kernarg = w.Kernarg;
        int tail = w.SRange(12);
        string residual = FinishText($"%s[{tail}:{tail + 1}]"), sum = FinishText($"%s[{tail + 2}:{tail + 3}]"), gain = FinishText($"%s[{tail + 4}:{tail + 5}]");
        string normalized = FinishText($"%s[{tail + 6}:{tail + 7}]"), ygain = FinishText($"%s[{tail + 8}:{tail + 9}]");
        int scalars = w.SRange(5);
        string eps = FinishText($"%s{scalars}"), offset = FinishText($"%s{scalars + 1}"), yeps = FinishText($"%s{scalars + 2}"), yoffset = FinishText($"%s{scalars + 3}");
        string flags = FinishText($"%s{scalars + 4}");
        w.L($"s_load_dwordx8 %s[{tail}:{tail + 7}], {kernarg}, 0x{AddNormTailOffset:x}      ; residual, sum, gain, normalized");
        w.L($"s_load_dwordx2 {ygain}, {kernarg}, 0x{AddNormTailOffset + 32:x}");
        w.L($"s_load_dwordx4 %s[{scalars}:{scalars + 3}], {kernarg}, 0x{AddNormTailOffset + 40:x}  ; eps, offset, yeps, yoffset");
        w.L($"s_load_dword {flags}, {kernarg}, 0x{AddNormTailOffset + 56:x}");
        string laneBytes = w.V(), rc = w.V(), row = w.S(), rowBytes = w.S(), bit = w.S();
        w.L($"v_lshlrev_b32_e32 {laneBytes}, 2, {w.Tid}");
        w.L($"v_cvt_f32_u32_e32 {rc}, {n}");
        w.L($"v_rcp_f32_e32 {rc}, {rc}");
        w.L($"s_lshl_b32 {rowBytes}, {n}, 2");
        string yb = w.SPair(), rb = w.SPair(), sb = w.SPair(), nb = w.SPair();
        w.L($"s_mov_b64 {yb}, {y}");
        w.L($"s_mov_b64 {rb}, {residual}");
        w.L($"s_mov_b64 {sb}, {sum}");
        w.L($"s_mov_b64 {nb}, {normalized}");
        w.L($"s_mov_b32 {row}, 0");
        w.Label(".Ltail_row");
        w.L($"s_cmp_ge_u32 {row}, {m}");
        w.L($"s_cbranch_scc1 .Ldone");

        // Pass 1: Σ y² (y as the other blocks wrote it).
        const int Wide = 2, Narrow = 1;
        var acc = new string[WaveThreads];
        for (int j = 0; j < WaveThreads; j++)
        {
            acc[j] = w.V();
            w.L($"v_mov_b32_e32 {acc[j]}, 0");
        }

        var vs = Registers(w, Wide * WaveThreads);
        WaveColumns(w, n, laneBytes, Wide,
            (e, off, imm) => FinishText($"global_load_dword {vs[e]}, {off}, {yb}{imm} glc"),
            (e, _, _) => FinishText($"v_fma_f32 {acc[e % WaveThreads]}, {vs[e]}, {vs[e]}, {acc[e % WaveThreads]}"), vs);
        string total = WaveReduceAdd(w, acc, laneBytes);
        string ry = w.V();
        w.L($"v_mul_f32_e32 {ry}, {total}, {rc}");
        w.L($"v_add_f32_e32 {ry}, {yeps}, {ry}");
        w.L($"v_rsq_f32_e32 {ry}, {ry}");

        // Pass 2: y normalized (flags bit 0), sum = residual + y, Σ sum².
        var acc2 = new string[WaveThreads];
        for (int j = 0; j < WaveThreads; j++)
        {
            acc2[j] = w.V();
        }

        var ys = Registers(w, Narrow * WaveThreads);
        var rs = Registers(w, Narrow * WaveThreads);
        var gs = Registers(w, Narrow * WaveThreads);
        var sums = Registers(w, Narrow * WaveThreads);
        void Pass2(bool normalize)
        {
            for (int j = 0; j < WaveThreads; j++)
            {
                w.L($"v_mov_b32_e32 {acc2[j]}, 0");
            }

            WaveColumns(w, n, laneBytes, Narrow,
                (e, off, imm) => FinishText($"global_load_dword {ys[e]}, {off}, {yb}{imm} glc\nglobal_load_dword {rs[e]}, {off}, {rb}{imm}")
                                 + (normalize ? FinishText($"\nglobal_load_dword {gs[e]}, {off}, {ygain}{imm}") : ""),
                (e, off, imm) => (normalize
                    ? FinishText($"""
                        v_add_f32_e32 {gs[e]}, {yoffset}, {gs[e]}
                        v_mul_f32_e32 {ys[e]}, {ys[e]}, {ry}
                        v_mul_f32_e32 {ys[e]}, {ys[e]}, {gs[e]}
                        global_store_dword {off}, {ys[e]}, {yb}{imm}

                        """)
                    : "") + FinishText($"""
                    v_add_f32_e32 {sums[e]}, {rs[e]}, {ys[e]}
                    global_store_dword {off}, {sums[e]}, {sb}{imm}
                    v_fma_f32 {acc2[e % WaveThreads]}, {sums[e]}, {sums[e]}, {acc2[e % WaveThreads]}
                    """), [.. ys, .. rs, .. gs, .. sums]);
        }

        w.L($"s_and_b32 {bit}, {flags}, 1");
        w.L($"s_cmp_eq_u32 {bit}, 0");
        w.L($"s_cbranch_scc1 .Ltail_plain");
        Pass2(normalize: true);
        w.L($"s_branch .Ltail_added");
        w.Label(".Ltail_plain");
        Pass2(normalize: false);
        w.Label(".Ltail_added");
        string total2 = WaveReduceAdd(w, acc2, laneBytes);
        string r = w.V();
        w.L($"v_mul_f32_e32 {r}, {total2}, {rc}");
        w.L($"v_add_f32_e32 {r}, {eps}, {r}");
        w.L($"v_rsq_f32_e32 {r}, {r}");

        // Pass 3: normalized = sum · r · (gain + offset), sum read back by the lanes that wrote it.
        var outs = Registers(w, Narrow * WaveThreads);
        WaveColumns(w, n, laneBytes, Narrow,
            (e, off, imm) => FinishText($"global_load_dword {sums[e]}, {off}, {sb}{imm}\nglobal_load_dword {gs[e]}, {off}, {gain}{imm}"),
            (e, off, imm) => FinishText($"""
                v_add_f32_e32 {gs[e]}, {offset}, {gs[e]}
                v_mul_f32_e32 {outs[e]}, {sums[e]}, {r}
                v_mul_f32_e32 {outs[e]}, {outs[e]}, {gs[e]}
                global_store_dword {off}, {outs[e]}, {nb}{imm}
                """), [.. sums, .. gs, .. outs]);
        foreach (string pointer in (string[])[yb, rb, sb, nb])
        {
            w.L($"s_add_u32 {ProductWriter.Lo(pointer)}, {ProductWriter.Lo(pointer)}, {rowBytes}");
            w.L($"s_addc_u32 {ProductWriter.Hi(pointer)}, {ProductWriter.Hi(pointer)}, 0");
        }

        w.L($"s_add_u32 {row}, {row}, 1");
        w.L($"s_branch .Ltail_row");
    }

    /// <summary>
    /// A multi-output product's gate/up pair (act, at <see cref="MultiPairOffset"/>, not 0): every block of either product
    /// whose columns are final arrives at counters[2 · blocks + column range]; the second writes hidden = act(y0) · y1 for
    /// the range's columns (gated_act_f32's formulas, act − 1 its kind), from y0 and y1 as their blocks stored them (glc),
    /// as gemv_multi_finish_f32 computes it from the same values.
    /// </summary>
    private static void PairFinish(ProductWriter w, string m, string counters, string[] columns, int cpw)
    {
        w.Label(".Lpair");
        string kernarg = w.Kernarg, act = w.S(), index = w.S();
        w.L($"s_load_dword {act}, {kernarg}, 0x{MultiPairOffset:x}             ; act: 0 no pair");
        w.L($"s_cmp_eq_u32 {act}, 0");
        w.L($"s_cbranch_scc1 .Ldone");
        w.L($"s_cmp_eq_u64 {counters}, 0");
        w.L($"s_cbranch_scc1 .Ldone");
        w.L($"s_lshl_b32 {index}, {w.Scalar("blocks")}, 1");
        w.L($"s_add_u32 {index}, {index}, {w.GroupX}                     ; the pair's counter of this column range");
        WaveArrive(w, counters, index, "2", ".Ldone");
        string y0 = w.SPair(), y1 = w.SPair(), hidden = w.SPair(), n0 = w.S();
        w.L($"s_load_dwordx2 {y0}, {kernarg}, 0x{MultiGemvSharedBytes + 16:x}");
        w.L($"s_load_dword {n0}, {kernarg}, 0x{MultiGemvSharedBytes + 32:x}");
        w.L($"s_load_dwordx2 {y1}, {kernarg}, 0x{MultiGemvSharedBytes + MultiGemvProductBytes + 16:x}");
        w.L($"s_load_dwordx2 {hidden}, {kernarg}, 0x{MultiPairOffset + 4:x}");

        // The activations, one text a kind, each with its constants set in its own branch (a kind's alone are live: the
        // kernel's register count stays near the product's).
        string g = w.V(), u = w.V(), h = w.V();
        var kinds = new System.Text.StringBuilder[GatedKinds.Length];
        for (int kind = 0; kind < kinds.Length; kind++)
        {
            var p = new PointwiseWriter(firstVector: 3000 + 200 * kind, firstScalar: 3000 + 200 * kind);
            var text = new System.Text.StringBuilder();
            p.Into(text);
            p.Gated(h, g, u, kind);
            kinds[kind] = new System.Text.StringBuilder().Append(p.Held).Append(text);
        }

        string row = w.S(), rowFirst = w.S(), c = w.S(), column = w.V(), at = w.V(), saved = w.SPair();
        w.L($"s_mov_b32 {row}, 0");
        w.Label(".Lpair_row");
        w.L($"s_cmp_ge_u32 {row}, {m}");
        w.L($"s_cbranch_scc1 .Ldone");
        w.L($"s_mul_i32 {rowFirst}, {row}, {n0}");
        w.L($"s_mov_b32 {c}, 0");
        w.Label(".Lpair_column");
        w.L($"s_cmp_ge_u32 {c}, {cpw}");
        w.L($"s_cbranch_scc1 .Lpair_next");
        // Every lane computes (its column clamped into the row: registers written under the whole EXEC); the lanes past n0
        // store nothing.
        string lastColumn = w.S(), clamped = w.V(), within = w.SPair();
        w.L($"s_sub_u32 {lastColumn}, {n0}, 1");
        w.L($"v_add_u32_e32 {column}, {c}, {columns[0]}");
        w.L($"v_cmp_gt_u32_e64 {within}, {n0}, {column}");
        w.L($"v_min_u32_e32 {clamped}, {lastColumn}, {column}");
        w.L($"v_add_u32_e32 {at}, {rowFirst}, {clamped}");
        w.L($"v_lshlrev_b32_e32 {at}, 2, {at}");
        w.L($"global_load_dword {g}, {at}, {y0} glc");
        w.L($"global_load_dword {u}, {at}, {y1} glc");
        for (int kind = 0; kind < kinds.Length; kind++)
        {
            w.L($"s_cmp_eq_u32 {act}, {kind + 1}");
            w.L($"s_cbranch_scc0 .Lpair_kind{kind}");
            w.Lines(kinds[kind]);
            w.L($"s_and_saveexec_b64 {saved}, {within}");
            w.L($"global_store_dword {at}, {h}, {hidden}");
            w.L($"s_or_b64 exec, exec, {saved}");
            w.L($"s_branch .Lpair_skip");
            w.Label($".Lpair_kind{kind}");
        }

        w.Label(".Lpair_skip");
        w.L($"s_add_u32 {c}, {c}, 1");
        w.L($"s_branch .Lpair_column");
        w.Label(".Lpair_next");
        w.L($"s_add_u32 {row}, {row}, 1");
        w.L($"s_branch .Lpair_row");
    }

    private static string[] Registers(ProductWriter w, int count)
    {
        var registers = new string[count];
        for (int i = 0; i < count; i++)
        {
            registers[i] = w.V();
        }

        return registers;
    }
}
