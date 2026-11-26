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
// The memory order is LLVM's AMDGPU memory model at agent scope, written once for every family and expanded by the
// assembler as ptxas expands PTX's: a block's stores complete (s_waitcnt vmcnt(0) vscnt(0), written out: no register
// waits for a store; gfx9 counts stores on vmcnt, ISA 10 and 11 add s_waitcnt_vscnt null, 0) before the atomic, which
// returns the count (glc) to lane 0; the last block reads what the others stored with device-scope loads
// (scope:SCOPE_DEV: glc on gfx9 and ISA 11, glc dlc on ISA 10), which miss the caches a compute unit's waves share and
// read the L2 every compute unit shares (CUDA's ld.global.cg). The interpreter checks each step: a load of another
// block's store without the atomic between them, before the stores were retired, or without the family's device scope,
// is reported as a race.
internal static partial class GcnKernels
{
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
            s_waitcnt vmcnt(0) vscnt(0)
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
}
