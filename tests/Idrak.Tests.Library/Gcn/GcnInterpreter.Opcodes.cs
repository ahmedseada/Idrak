// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Tests.Gcn;

// Made by tests/Idrak.Tests/data/gcn/interpreter/make_opcode_table.py from LLVM 18.1.3's disassembler; do not edit, rerun the
// script. One opcode a line: the encoding, the opcode (hex), the CPUs that have it (gfx900, gfx906, gfx908, gfx90c, 1 or 0 each), the mnemonic
// as LLVM prints it, and the operand template in print order (a field, its register class and width; '=' a fixed
// operand; K an immediate and its kind; O the offset's kind); after '|', FLAT's template when glc is set.
internal static partial class GcnOpcodeTable
{
    internal const string Cpus = "gfx900 gfx906 gfx908 gfx90c";

    internal const string Text = """
        sop1 000 1111 s_mov_b32 D:s1 S0:s1
        sop1 001 1111 s_mov_b64 D:s2 S0:s2
        sop1 002 1111 s_cmov_b32 D:s1 S0:s1
        sop1 003 1111 s_cmov_b64 D:s2 S0:s2
        sop1 004 1111 s_not_b32 D:s1 S0:s1
        sop1 005 1111 s_not_b64 D:s2 S0:s2
        sop1 006 1111 s_wqm_b32 D:s1 S0:s1
        sop1 007 1111 s_wqm_b64 D:s2 S0:s2
        sop1 008 1111 s_brev_b32 D:s1 S0:s1
        sop1 009 1111 s_brev_b64 D:s2 S0:s2
        sop1 00a 1111 s_bcnt0_i32_b32 D:s1 S0:s1
        sop1 00b 1111 s_bcnt0_i32_b64 D:s1 S0:s2
        sop1 00c 1111 s_bcnt1_i32_b32 D:s1 S0:s1
        sop1 00d 1111 s_bcnt1_i32_b64 D:s1 S0:s2
        sop1 00e 1111 s_ff0_i32_b32 D:s1 S0:s1
        sop1 00f 1111 s_ff0_i32_b64 D:s1 S0:s2
        sop1 010 1111 s_ff1_i32_b32 D:s1 S0:s1
        sop1 011 1111 s_ff1_i32_b64 D:s1 S0:s2
        sop1 012 1111 s_flbit_i32_b32 D:s1 S0:s1
        sop1 013 1111 s_flbit_i32_b64 D:s1 S0:s2
        sop1 014 1111 s_flbit_i32 D:s1 S0:s1
        sop1 015 1111 s_flbit_i32_i64 D:s1 S0:s2
        sop1 016 1111 s_sext_i32_i8 D:s1 S0:s1
        sop1 017 1111 s_sext_i32_i16 D:s1 S0:s1
        sop1 018 1111 s_bitset0_b32 D:s1 S0:s1
        sop1 019 1111 s_bitset0_b64 D:s2 S0:s1
        sop1 01a 1111 s_bitset1_b32 D:s1 S0:s1
        sop1 01b 1111 s_bitset1_b64 D:s2 S0:s1
        sop1 01c 1111 s_getpc_b64 D:s2
        sop1 01d 1111 s_setpc_b64 S0:s2
        sop1 01e 1111 s_swappc_b64 D:s2 S0:s2
        sop1 01f 1111 s_rfe_b64 S0:s2
        sop1 020 1111 s_and_saveexec_b64 D:s2 S0:s2
        sop1 021 1111 s_or_saveexec_b64 D:s2 S0:s2
        sop1 022 1111 s_xor_saveexec_b64 D:s2 S0:s2
        sop1 023 1111 s_andn2_saveexec_b64 D:s2 S0:s2
        sop1 024 1111 s_orn2_saveexec_b64 D:s2 S0:s2
        sop1 025 1111 s_nand_saveexec_b64 D:s2 S0:s2
        sop1 026 1111 s_nor_saveexec_b64 D:s2 S0:s2
        sop1 027 1111 s_xnor_saveexec_b64 D:s2 S0:s2
        sop1 028 1111 s_quadmask_b32 D:s1 S0:s1
        sop1 029 1111 s_quadmask_b64 D:s2 S0:s2
        sop1 02a 1111 s_movrels_b32 D:s1 S0:s1
        sop1 02b 1111 s_movrels_b64 D:s2 S0:s2
        sop1 02c 1111 s_movreld_b32 D:s1 S0:s1
        sop1 02d 1111 s_movreld_b64 D:s2 S0:s2
        sop1 02e 1111 s_cbranch_join S0:s1
        sop1 030 1111 s_abs_i32 D:s1 S0:s1
        sop1 032 1111 s_set_gpr_idx_idx S0:s1
        sop1 033 1111 s_andn1_saveexec_b64 D:s2 S0:s2
        sop1 034 1111 s_orn1_saveexec_b64 D:s2 S0:s2
        sop1 035 1111 s_andn1_wrexec_b64 D:s2 S0:s2
        sop1 036 1111 s_andn2_wrexec_b64 D:s2 S0:s2
        sop1 037 1111 s_bitreplicate_b64_b32 D:s2 S0:s1
        sop2 000 1111 s_add_u32 D:s1 S0:s1 S1:s1
        sop2 001 1111 s_sub_u32 D:s1 S0:s1 S1:s1
        sop2 002 1111 s_add_i32 D:s1 S0:s1 S1:s1
        sop2 003 1111 s_sub_i32 D:s1 S0:s1 S1:s1
        sop2 004 1111 s_addc_u32 D:s1 S0:s1 S1:s1
        sop2 005 1111 s_subb_u32 D:s1 S0:s1 S1:s1
        sop2 006 1111 s_min_i32 D:s1 S0:s1 S1:s1
        sop2 007 1111 s_min_u32 D:s1 S0:s1 S1:s1
        sop2 008 1111 s_max_i32 D:s1 S0:s1 S1:s1
        sop2 009 1111 s_max_u32 D:s1 S0:s1 S1:s1
        sop2 00a 1111 s_cselect_b32 D:s1 S0:s1 S1:s1
        sop2 00b 1111 s_cselect_b64 D:s2 S0:s2 S1:s2
        sop2 00c 1111 s_and_b32 D:s1 S0:s1 S1:s1
        sop2 00d 1111 s_and_b64 D:s2 S0:s2 S1:s2
        sop2 00e 1111 s_or_b32 D:s1 S0:s1 S1:s1
        sop2 00f 1111 s_or_b64 D:s2 S0:s2 S1:s2
        sop2 010 1111 s_xor_b32 D:s1 S0:s1 S1:s1
        sop2 011 1111 s_xor_b64 D:s2 S0:s2 S1:s2
        sop2 012 1111 s_andn2_b32 D:s1 S0:s1 S1:s1
        sop2 013 1111 s_andn2_b64 D:s2 S0:s2 S1:s2
        sop2 014 1111 s_orn2_b32 D:s1 S0:s1 S1:s1
        sop2 015 1111 s_orn2_b64 D:s2 S0:s2 S1:s2
        sop2 016 1111 s_nand_b32 D:s1 S0:s1 S1:s1
        sop2 017 1111 s_nand_b64 D:s2 S0:s2 S1:s2
        sop2 018 1111 s_nor_b32 D:s1 S0:s1 S1:s1
        sop2 019 1111 s_nor_b64 D:s2 S0:s2 S1:s2
        sop2 01a 1111 s_xnor_b32 D:s1 S0:s1 S1:s1
        sop2 01b 1111 s_xnor_b64 D:s2 S0:s2 S1:s2
        sop2 01c 1111 s_lshl_b32 D:s1 S0:s1 S1:s1
        sop2 01d 1111 s_lshl_b64 D:s2 S0:s2 S1:s1
        sop2 01e 1111 s_lshr_b32 D:s1 S0:s1 S1:s1
        sop2 01f 1111 s_lshr_b64 D:s2 S0:s2 S1:s1
        sop2 020 1111 s_ashr_i32 D:s1 S0:s1 S1:s1
        sop2 021 1111 s_ashr_i64 D:s2 S0:s2 S1:s1
        sop2 022 1111 s_bfm_b32 D:s1 S0:s1 S1:s1
        sop2 023 1111 s_bfm_b64 D:s2 S0:s1 S1:s1
        sop2 024 1111 s_mul_i32 D:s1 S0:s1 S1:s1
        sop2 025 1111 s_bfe_u32 D:s1 S0:s1 S1:s1
        sop2 026 1111 s_bfe_i32 D:s1 S0:s1 S1:s1
        sop2 027 1111 s_bfe_u64 D:s2 S0:s2 S1:s1
        sop2 028 1111 s_bfe_i64 D:s2 S0:s2 S1:s1
        sop2 029 1111 s_cbranch_g_fork S0:s2 S1:s2
        sop2 02a 1111 s_absdiff_i32 D:s1 S0:s1 S1:s1
        sop2 02b 1111 s_rfe_restore_b64 S0:s2 S1:s1
        sop2 02c 1111 s_mul_hi_u32 D:s1 S0:s1 S1:s1
        sop2 02d 1111 s_mul_hi_i32 D:s1 S0:s1 S1:s1
        sop2 02e 1111 s_lshl1_add_u32 D:s1 S0:s1 S1:s1
        sop2 02f 1111 s_lshl2_add_u32 D:s1 S0:s1 S1:s1
        sop2 030 1111 s_lshl3_add_u32 D:s1 S0:s1 S1:s1
        sop2 031 1111 s_lshl4_add_u32 D:s1 S0:s1 S1:s1
        sop2 032 1111 s_pack_ll_b32_b16 D:s1 S0:s1 S1:s1
        sop2 033 1111 s_pack_lh_b32_b16 D:s1 S0:s1 S1:s1
        sop2 034 1111 s_pack_hh_b32_b16 D:s1 S0:s1 S1:s1
        sopk 000 1111 s_movk_i32 D:s1 K:hex16
        sopk 001 1111 s_cmovk_i32 D:s1 K:hex16
        sopk 002 1111 s_cmpk_eq_i32 D:s1 K:hex16
        sopk 003 1111 s_cmpk_lg_i32 D:s1 K:hex16
        sopk 004 1111 s_cmpk_gt_i32 D:s1 K:hex16
        sopk 005 1111 s_cmpk_ge_i32 D:s1 K:hex16
        sopk 006 1111 s_cmpk_lt_i32 D:s1 K:hex16
        sopk 007 1111 s_cmpk_le_i32 D:s1 K:hex16
        sopk 008 1111 s_cmpk_eq_u32 D:s1 K:hex16
        sopk 009 1111 s_cmpk_lg_u32 D:s1 K:hex16
        sopk 00a 1111 s_cmpk_gt_u32 D:s1 K:hex16
        sopk 00b 1111 s_cmpk_ge_u32 D:s1 K:hex16
        sopk 00c 1111 s_cmpk_lt_u32 D:s1 K:hex16
        sopk 00d 1111 s_cmpk_le_u32 D:s1 K:hex16
        sopk 00e 1111 s_addk_i32 D:s1 K:hex16
        sopk 00f 1111 s_mulk_i32 D:s1 K:hex16
        sopk 010 1111 s_cbranch_i_fork D:s2 K:dec
        sopk 011 1111 s_getreg_b32 D:s1 K:hwreg
        sopk 012 1111 s_setreg_b32 K:hwreg D:s1
        sopk 014 1111 s_setreg_imm32_b32 K:hwreg K:lit
        sopk 015 1111 s_call_b64 D:s2 K:dec
        sopc 000 1111 s_cmp_eq_i32 S0:s1 S1:s1
        sopc 001 1111 s_cmp_lg_i32 S0:s1 S1:s1
        sopc 002 1111 s_cmp_gt_i32 S0:s1 S1:s1
        sopc 003 1111 s_cmp_ge_i32 S0:s1 S1:s1
        sopc 004 1111 s_cmp_lt_i32 S0:s1 S1:s1
        sopc 005 1111 s_cmp_le_i32 S0:s1 S1:s1
        sopc 006 1111 s_cmp_eq_u32 S0:s1 S1:s1
        sopc 007 1111 s_cmp_lg_u32 S0:s1 S1:s1
        sopc 008 1111 s_cmp_gt_u32 S0:s1 S1:s1
        sopc 009 1111 s_cmp_ge_u32 S0:s1 S1:s1
        sopc 00a 1111 s_cmp_lt_u32 S0:s1 S1:s1
        sopc 00b 1111 s_cmp_le_u32 S0:s1 S1:s1
        sopc 00c 1111 s_bitcmp0_b32 S0:s1 S1:s1
        sopc 00d 1111 s_bitcmp1_b32 S0:s1 S1:s1
        sopc 00e 1111 s_bitcmp0_b64 S0:s2 S1:s1
        sopc 00f 1111 s_bitcmp1_b64 S0:s2 S1:s1
        sopc 010 1111 s_setvskip S0:s1 S1:s1
        sopc 011 1111 s_set_gpr_idx_on S0:s1 S1:gpridx
        sopc 012 1111 s_cmp_eq_u64 S0:s2 S1:s2
        sopc 013 1111 s_cmp_lg_u64 S0:s2 S1:s2
        sopp 000 1111 s_nop K:dec
        sopp 001 1111 s_endpgm K:dec
        sopp 002 1111 s_branch K:dec
        sopp 003 1111 s_wakeup -
        sopp 004 1111 s_cbranch_scc0 K:dec
        sopp 005 1111 s_cbranch_scc1 K:dec
        sopp 006 1111 s_cbranch_vccz K:dec
        sopp 007 1111 s_cbranch_vccnz K:dec
        sopp 008 1111 s_cbranch_execz K:dec
        sopp 009 1111 s_cbranch_execnz K:dec
        sopp 00a 1111 s_barrier -
        sopp 00b 1111 s_setkill K:dec
        sopp 00c 1111 s_waitcnt K:waitcnt
        sopp 00d 1111 s_sethalt K:dec
        sopp 00e 1111 s_sleep K:dec
        sopp 00f 1111 s_setprio K:dec
        sopp 010 1111 s_sendmsg K:sendmsg
        sopp 011 1111 s_sendmsghalt K:sendmsg
        sopp 012 1111 s_trap K:dec
        sopp 013 1111 s_icache_inv -
        sopp 014 1111 s_incperflevel K:dec
        sopp 015 1111 s_decperflevel K:dec
        sopp 016 1111 s_ttracedata -
        sopp 017 1111 s_cbranch_cdbgsys K:dec
        sopp 018 1111 s_cbranch_cdbguser K:dec
        sopp 019 1111 s_cbranch_cdbgsys_or_user K:dec
        sopp 01a 1111 s_cbranch_cdbgsys_and_user K:dec
        sopp 01b 1111 s_endpgm_saved -
        sopp 01c 1111 s_set_gpr_idx_off -
        sopp 01d 1111 s_set_gpr_idx_mode S1:gpridx
        sopp 01e 1111 s_endpgm_ordered_ps_done -
        smem 000 1111 s_load_dword D:s1 B:s2 O:hex
        smem 001 1111 s_load_dwordx2 D:s2 B:s2 O:hex
        smem 002 1111 s_load_dwordx4 D:s4 B:s2 O:hex
        smem 003 1111 s_load_dwordx8 D:s8 B:s2 O:hex
        smem 004 1111 s_load_dwordx16 D:s16 B:s2 O:hex
        smem 005 1111 s_scratch_load_dword D:s1 B:s2 O:hex
        smem 006 1111 s_scratch_load_dwordx2 D:s2 B:s2 O:hex
        smem 007 1111 s_scratch_load_dwordx4 D:s4 B:s2 O:hex
        smem 008 1111 s_buffer_load_dword D:s1 B:s4 O:hex
        smem 009 1111 s_buffer_load_dwordx2 D:s2 B:s4 O:hex
        smem 00a 1111 s_buffer_load_dwordx4 D:s4 B:s4 O:hex
        smem 00b 1111 s_buffer_load_dwordx8 D:s8 B:s4 O:hex
        smem 00c 1111 s_buffer_load_dwordx16 D:s16 B:s4 O:hex
        smem 010 1111 s_store_dword D:s1 B:s2 O:hex
        smem 011 1111 s_store_dwordx2 D:s2 B:s2 O:hex
        smem 012 1111 s_store_dwordx4 D:s4 B:s2 O:hex
        smem 015 1111 s_scratch_store_dword D:s1 B:s2 O:hex
        smem 016 1111 s_scratch_store_dwordx2 D:s2 B:s2 O:hex
        smem 017 1111 s_scratch_store_dwordx4 D:s4 B:s2 O:hex
        smem 018 1111 s_buffer_store_dword D:s1 B:s4 O:hex
        smem 019 1111 s_buffer_store_dwordx2 D:s2 B:s4 O:hex
        smem 01a 1111 s_buffer_store_dwordx4 D:s4 B:s4 O:hex
        smem 020 1111 s_dcache_inv -
        smem 021 1111 s_dcache_wb -
        smem 022 1111 s_dcache_inv_vol -
        smem 023 1111 s_dcache_wb_vol -
        smem 024 1111 s_memtime D:s2
        smem 025 1111 s_memrealtime D:s2
        smem 026 1111 s_atc_probe K:dec B:s2 O:hex
        smem 027 1111 s_atc_probe_buffer K:dec B:s4 O:hex
        smem 028 1111 s_dcache_discard B:s2 O:hex
        smem 029 1111 s_dcache_discard_x2 B:s2 O:hex
        smem 040 1111 s_buffer_atomic_swap D:s1 B:s4 O:hex
        smem 041 1111 s_buffer_atomic_cmpswap D:s2 B:s4 O:hex
        smem 042 1111 s_buffer_atomic_add D:s1 B:s4 O:hex
        smem 043 1111 s_buffer_atomic_sub D:s1 B:s4 O:hex
        smem 044 1111 s_buffer_atomic_smin D:s1 B:s4 O:hex
        smem 045 1111 s_buffer_atomic_umin D:s1 B:s4 O:hex
        smem 046 1111 s_buffer_atomic_smax D:s1 B:s4 O:hex
        smem 047 1111 s_buffer_atomic_umax D:s1 B:s4 O:hex
        smem 048 1111 s_buffer_atomic_and D:s1 B:s4 O:hex
        smem 049 1111 s_buffer_atomic_or D:s1 B:s4 O:hex
        smem 04a 1111 s_buffer_atomic_xor D:s1 B:s4 O:hex
        smem 04b 1111 s_buffer_atomic_inc D:s1 B:s4 O:hex
        smem 04c 1111 s_buffer_atomic_dec D:s1 B:s4 O:hex
        smem 060 1111 s_buffer_atomic_swap_x2 D:s2 B:s4 O:hex
        smem 061 1111 s_buffer_atomic_cmpswap_x2 D:s4 B:s4 O:hex
        smem 062 1111 s_buffer_atomic_add_x2 D:s2 B:s4 O:hex
        smem 063 1111 s_buffer_atomic_sub_x2 D:s2 B:s4 O:hex
        smem 064 1111 s_buffer_atomic_smin_x2 D:s2 B:s4 O:hex
        smem 065 1111 s_buffer_atomic_umin_x2 D:s2 B:s4 O:hex
        smem 066 1111 s_buffer_atomic_smax_x2 D:s2 B:s4 O:hex
        smem 067 1111 s_buffer_atomic_umax_x2 D:s2 B:s4 O:hex
        smem 068 1111 s_buffer_atomic_and_x2 D:s2 B:s4 O:hex
        smem 069 1111 s_buffer_atomic_or_x2 D:s2 B:s4 O:hex
        smem 06a 1111 s_buffer_atomic_xor_x2 D:s2 B:s4 O:hex
        smem 06b 1111 s_buffer_atomic_inc_x2 D:s2 B:s4 O:hex
        smem 06c 1111 s_buffer_atomic_dec_x2 D:s2 B:s4 O:hex
        smem 080 1111 s_atomic_swap D:s1 B:s2 O:hex
        smem 081 1111 s_atomic_cmpswap D:s2 B:s2 O:hex
        smem 082 1111 s_atomic_add D:s1 B:s2 O:hex
        smem 083 1111 s_atomic_sub D:s1 B:s2 O:hex
        smem 084 1111 s_atomic_smin D:s1 B:s2 O:hex
        smem 085 1111 s_atomic_umin D:s1 B:s2 O:hex
        smem 086 1111 s_atomic_smax D:s1 B:s2 O:hex
        smem 087 1111 s_atomic_umax D:s1 B:s2 O:hex
        smem 088 1111 s_atomic_and D:s1 B:s2 O:hex
        smem 089 1111 s_atomic_or D:s1 B:s2 O:hex
        smem 08a 1111 s_atomic_xor D:s1 B:s2 O:hex
        smem 08b 1111 s_atomic_inc D:s1 B:s2 O:hex
        smem 08c 1111 s_atomic_dec D:s1 B:s2 O:hex
        smem 0a0 1111 s_atomic_swap_x2 D:s2 B:s2 O:hex
        smem 0a1 1111 s_atomic_cmpswap_x2 D:s4 B:s2 O:hex
        smem 0a2 1111 s_atomic_add_x2 D:s2 B:s2 O:hex
        smem 0a3 1111 s_atomic_sub_x2 D:s2 B:s2 O:hex
        smem 0a4 1111 s_atomic_smin_x2 D:s2 B:s2 O:hex
        smem 0a5 1111 s_atomic_umin_x2 D:s2 B:s2 O:hex
        smem 0a6 1111 s_atomic_smax_x2 D:s2 B:s2 O:hex
        smem 0a7 1111 s_atomic_umax_x2 D:s2 B:s2 O:hex
        smem 0a8 1111 s_atomic_and_x2 D:s2 B:s2 O:hex
        smem 0a9 1111 s_atomic_or_x2 D:s2 B:s2 O:hex
        smem 0aa 1111 s_atomic_xor_x2 D:s2 B:s2 O:hex
        smem 0ab 1111 s_atomic_inc_x2 D:s2 B:s2 O:hex
        smem 0ac 1111 s_atomic_dec_x2 D:s2 B:s2 O:hex
        vop1 000 1111 v_nop -
        vop1 001 1111 v_mov_b32_e32 D:v1 S0:v1
        vop1 002 1111 v_readfirstlane_b32 D:s1 S0:v1
        vop1 003 1111 v_cvt_i32_f64_e32 D:v1 S0:v2
        vop1 004 1111 v_cvt_f64_i32_e32 D:v2 S0:v1
        vop1 005 1111 v_cvt_f32_i32_e32 D:v1 S0:v1
        vop1 006 1111 v_cvt_f32_u32_e32 D:v1 S0:v1
        vop1 007 1111 v_cvt_u32_f32_e32 D:v1 S0:v1
        vop1 008 1111 v_cvt_i32_f32_e32 D:v1 S0:v1
        vop1 00a 1111 v_cvt_f16_f32_e32 D:v1 S0:v1
        vop1 00b 1111 v_cvt_f32_f16_e32 D:v1 S0:v1
        vop1 00c 1111 v_cvt_rpi_i32_f32_e32 D:v1 S0:v1
        vop1 00d 1111 v_cvt_flr_i32_f32_e32 D:v1 S0:v1
        vop1 00e 1111 v_cvt_off_f32_i4_e32 D:v1 S0:v1
        vop1 00f 1111 v_cvt_f32_f64_e32 D:v1 S0:v2
        vop1 010 1111 v_cvt_f64_f32_e32 D:v2 S0:v1
        vop1 011 1111 v_cvt_f32_ubyte0_e32 D:v1 S0:v1
        vop1 012 1111 v_cvt_f32_ubyte1_e32 D:v1 S0:v1
        vop1 013 1111 v_cvt_f32_ubyte2_e32 D:v1 S0:v1
        vop1 014 1111 v_cvt_f32_ubyte3_e32 D:v1 S0:v1
        vop1 015 1111 v_cvt_u32_f64_e32 D:v1 S0:v2
        vop1 016 1111 v_cvt_f64_u32_e32 D:v2 S0:v1
        vop1 017 1111 v_trunc_f64_e32 D:v2 S0:v2
        vop1 018 1111 v_ceil_f64_e32 D:v2 S0:v2
        vop1 019 1111 v_rndne_f64_e32 D:v2 S0:v2
        vop1 01a 1111 v_floor_f64_e32 D:v2 S0:v2
        vop1 01b 1111 v_fract_f32_e32 D:v1 S0:v1
        vop1 01c 1111 v_trunc_f32_e32 D:v1 S0:v1
        vop1 01d 1111 v_ceil_f32_e32 D:v1 S0:v1
        vop1 01e 1111 v_rndne_f32_e32 D:v1 S0:v1
        vop1 01f 1111 v_floor_f32_e32 D:v1 S0:v1
        vop1 020 1111 v_exp_f32_e32 D:v1 S0:v1
        vop1 021 1111 v_log_f32_e32 D:v1 S0:v1
        vop1 022 1111 v_rcp_f32_e32 D:v1 S0:v1
        vop1 023 1111 v_rcp_iflag_f32_e32 D:v1 S0:v1
        vop1 024 1111 v_rsq_f32_e32 D:v1 S0:v1
        vop1 025 1111 v_rcp_f64_e32 D:v2 S0:v2
        vop1 026 1111 v_rsq_f64_e32 D:v2 S0:v2
        vop1 027 1111 v_sqrt_f32_e32 D:v1 S0:v1
        vop1 028 1111 v_sqrt_f64_e32 D:v2 S0:v2
        vop1 029 1111 v_sin_f32_e32 D:v1 S0:v1
        vop1 02a 1111 v_cos_f32_e32 D:v1 S0:v1
        vop1 02b 1111 v_not_b32_e32 D:v1 S0:v1
        vop1 02c 1111 v_bfrev_b32_e32 D:v1 S0:v1
        vop1 02d 1111 v_ffbh_u32_e32 D:v1 S0:v1
        vop1 02e 1111 v_ffbl_b32_e32 D:v1 S0:v1
        vop1 02f 1111 v_ffbh_i32_e32 D:v1 S0:v1
        vop1 030 1111 v_frexp_exp_i32_f64_e32 D:v1 S0:v2
        vop1 031 1111 v_frexp_mant_f64_e32 D:v2 S0:v2
        vop1 032 1111 v_fract_f64_e32 D:v2 S0:v2
        vop1 033 1111 v_frexp_exp_i32_f32_e32 D:v1 S0:v1
        vop1 034 1111 v_frexp_mant_f32_e32 D:v1 S0:v1
        vop1 035 1111 v_clrexcp -
        vop1 037 1111 v_screen_partition_4se_b32_e32 D:v1 S0:v1
        vop1 039 1111 v_cvt_f16_u16_e32 D:v1 S0:v1
        vop1 03a 1111 v_cvt_f16_i16_e32 D:v1 S0:v1
        vop1 03b 1111 v_cvt_u16_f16_e32 D:v1 S0:v1
        vop1 03c 1111 v_cvt_i16_f16_e32 D:v1 S0:v1
        vop1 03d 1111 v_rcp_f16_e32 D:v1 S0:v1
        vop1 03e 1111 v_sqrt_f16_e32 D:v1 S0:v1
        vop1 03f 1111 v_rsq_f16_e32 D:v1 S0:v1
        vop1 040 1111 v_log_f16_e32 D:v1 S0:v1
        vop1 041 1111 v_exp_f16_e32 D:v1 S0:v1
        vop1 042 1111 v_frexp_mant_f16_e32 D:v1 S0:v1
        vop1 043 1111 v_frexp_exp_i16_f16_e32 D:v1 S0:v1
        vop1 044 1111 v_floor_f16_e32 D:v1 S0:v1
        vop1 045 1111 v_ceil_f16_e32 D:v1 S0:v1
        vop1 046 1111 v_trunc_f16_e32 D:v1 S0:v1
        vop1 047 1111 v_rndne_f16_e32 D:v1 S0:v1
        vop1 048 1111 v_fract_f16_e32 D:v1 S0:v1
        vop1 049 1111 v_sin_f16_e32 D:v1 S0:v1
        vop1 04a 1111 v_cos_f16_e32 D:v1 S0:v1
        vop1 04b 1111 v_exp_legacy_f32_e32 D:v1 S0:v1
        vop1 04c 1111 v_log_legacy_f32_e32 D:v1 S0:v1
        vop1 04d 1111 v_cvt_norm_i16_f16_e32 D:v1 S0:v1
        vop1 04e 1111 v_cvt_norm_u16_f16_e32 D:v1 S0:v1
        vop1 04f 1111 v_sat_pk_u8_i16_e32 D:v1 S0:v1
        vop1 051 1111 v_swap_b32 D:v1 S0:v1
        vop2 000 1111 v_cndmask_b32_e32 D:v1 S0:v1 S1:v1 =vcc
        vop2 001 1111 v_add_f32_e32 D:v1 S0:v1 S1:v1
        vop2 002 1111 v_sub_f32_e32 D:v1 S0:v1 S1:v1
        vop2 003 1111 v_subrev_f32_e32 D:v1 S0:v1 S1:v1
        vop2 004 1111 v_mul_legacy_f32_e32 D:v1 S0:v1 S1:v1
        vop2 005 1111 v_mul_f32_e32 D:v1 S0:v1 S1:v1
        vop2 006 1111 v_mul_i32_i24_e32 D:v1 S0:v1 S1:v1
        vop2 007 1111 v_mul_hi_i32_i24_e32 D:v1 S0:v1 S1:v1
        vop2 008 1111 v_mul_u32_u24_e32 D:v1 S0:v1 S1:v1
        vop2 009 1111 v_mul_hi_u32_u24_e32 D:v1 S0:v1 S1:v1
        vop2 00a 1111 v_min_f32_e32 D:v1 S0:v1 S1:v1
        vop2 00b 1111 v_max_f32_e32 D:v1 S0:v1 S1:v1
        vop2 00c 1111 v_min_i32_e32 D:v1 S0:v1 S1:v1
        vop2 00d 1111 v_max_i32_e32 D:v1 S0:v1 S1:v1
        vop2 00e 1111 v_min_u32_e32 D:v1 S0:v1 S1:v1
        vop2 00f 1111 v_max_u32_e32 D:v1 S0:v1 S1:v1
        vop2 010 1111 v_lshrrev_b32_e32 D:v1 S0:v1 S1:v1
        vop2 011 1111 v_ashrrev_i32_e32 D:v1 S0:v1 S1:v1
        vop2 012 1111 v_lshlrev_b32_e32 D:v1 S0:v1 S1:v1
        vop2 013 1111 v_and_b32_e32 D:v1 S0:v1 S1:v1
        vop2 014 1111 v_or_b32_e32 D:v1 S0:v1 S1:v1
        vop2 015 1111 v_xor_b32_e32 D:v1 S0:v1 S1:v1
        vop2 016 1111 v_mac_f32_e32 D:v1 S0:v1 S1:v1
        vop2 017 1111 v_madmk_f32 D:v1 S0:v1 K:lit S1:v1
        vop2 018 1111 v_madak_f32 D:v1 S0:v1 S1:v1 K:lit
        vop2 019 1111 v_add_co_u32_e32 D:v1 =vcc S0:v1 S1:v1
        vop2 01a 1111 v_sub_co_u32_e32 D:v1 =vcc S0:v1 S1:v1
        vop2 01b 1111 v_subrev_co_u32_e32 D:v1 =vcc S0:v1 S1:v1
        vop2 01c 1111 v_addc_co_u32_e32 D:v1 =vcc S0:v1 S1:v1 =vcc
        vop2 01d 1111 v_subb_co_u32_e32 D:v1 =vcc S0:v1 S1:v1 =vcc
        vop2 01e 1111 v_subbrev_co_u32_e32 D:v1 =vcc S0:v1 S1:v1 =vcc
        vop2 01f 1111 v_add_f16_e32 D:v1 S0:v1 S1:v1
        vop2 020 1111 v_sub_f16_e32 D:v1 S0:v1 S1:v1
        vop2 021 1111 v_subrev_f16_e32 D:v1 S0:v1 S1:v1
        vop2 022 1111 v_mul_f16_e32 D:v1 S0:v1 S1:v1
        vop2 023 1111 v_mac_f16_e32 D:v1 S0:v1 S1:v1
        vop2 024 1111 v_madmk_f16 D:v1 S0:v1 K:lit S1:v1
        vop2 025 1111 v_madak_f16 D:v1 S0:v1 S1:v1 K:lit
        vop2 026 1111 v_add_u16_e32 D:v1 S0:v1 S1:v1
        vop2 027 1111 v_sub_u16_e32 D:v1 S0:v1 S1:v1
        vop2 028 1111 v_subrev_u16_e32 D:v1 S0:v1 S1:v1
        vop2 029 1111 v_mul_lo_u16_e32 D:v1 S0:v1 S1:v1
        vop2 02a 1111 v_lshlrev_b16_e32 D:v1 S0:v1 S1:v1
        vop2 02b 1111 v_lshrrev_b16_e32 D:v1 S0:v1 S1:v1
        vop2 02c 1111 v_ashrrev_i16_e32 D:v1 S0:v1 S1:v1
        vop2 02d 1111 v_max_f16_e32 D:v1 S0:v1 S1:v1
        vop2 02e 1111 v_min_f16_e32 D:v1 S0:v1 S1:v1
        vop2 02f 1111 v_max_u16_e32 D:v1 S0:v1 S1:v1
        vop2 030 1111 v_max_i16_e32 D:v1 S0:v1 S1:v1
        vop2 031 1111 v_min_u16_e32 D:v1 S0:v1 S1:v1
        vop2 032 1111 v_min_i16_e32 D:v1 S0:v1 S1:v1
        vop2 033 1111 v_ldexp_f16_e32 D:v1 S0:v1 S1:v1
        vop2 034 1111 v_add_u32_e32 D:v1 S0:v1 S1:v1
        vop2 035 1111 v_sub_u32_e32 D:v1 S0:v1 S1:v1
        vop2 036 1111 v_subrev_u32_e32 D:v1 S0:v1 S1:v1
        vop2 037 0010 v_dot2c_f32_f16_e32 D:v1 S0:v1 S1:v1
        vop2 038 0010 v_dot2c_i32_i16_e32 D:v1 S0:v1 S1:v1
        vop2 039 0010 v_dot4c_i32_i8_e32 D:v1 S0:v1 S1:v1
        vop2 03a 0010 v_dot8c_i32_i4_e32 D:v1 S0:v1 S1:v1
        vop2 03b 0110 v_fmac_f32_e32 D:v1 S0:v1 S1:v1
        vop2 03c 0010 v_pk_fmac_f16_e32 D:v1 S0:v1 S1:v1
        vop2 03d 0110 v_xnor_b32_e32 D:v1 S0:v1 S1:v1
        vopc 010 1111 v_cmp_class_f32_e32 =vcc S0:v1 S1:v1
        vopc 011 1111 v_cmpx_class_f32_e32 =vcc S0:v1 S1:v1
        vopc 012 1111 v_cmp_class_f64_e32 =vcc S0:v2 S1:v1
        vopc 013 1111 v_cmpx_class_f64_e32 =vcc S0:v2 S1:v1
        vopc 014 1111 v_cmp_class_f16_e32 =vcc S0:v1 S1:v1
        vopc 015 1111 v_cmpx_class_f16_e32 =vcc S0:v1 S1:v1
        vopc 020 1111 v_cmp_f_f16_e32 =vcc S0:v1 S1:v1
        vopc 021 1111 v_cmp_lt_f16_e32 =vcc S0:v1 S1:v1
        vopc 022 1111 v_cmp_eq_f16_e32 =vcc S0:v1 S1:v1
        vopc 023 1111 v_cmp_le_f16_e32 =vcc S0:v1 S1:v1
        vopc 024 1111 v_cmp_gt_f16_e32 =vcc S0:v1 S1:v1
        vopc 025 1111 v_cmp_lg_f16_e32 =vcc S0:v1 S1:v1
        vopc 026 1111 v_cmp_ge_f16_e32 =vcc S0:v1 S1:v1
        vopc 027 1111 v_cmp_o_f16_e32 =vcc S0:v1 S1:v1
        vopc 028 1111 v_cmp_u_f16_e32 =vcc S0:v1 S1:v1
        vopc 029 1111 v_cmp_nge_f16_e32 =vcc S0:v1 S1:v1
        vopc 02a 1111 v_cmp_nlg_f16_e32 =vcc S0:v1 S1:v1
        vopc 02b 1111 v_cmp_ngt_f16_e32 =vcc S0:v1 S1:v1
        vopc 02c 1111 v_cmp_nle_f16_e32 =vcc S0:v1 S1:v1
        vopc 02d 1111 v_cmp_neq_f16_e32 =vcc S0:v1 S1:v1
        vopc 02e 1111 v_cmp_nlt_f16_e32 =vcc S0:v1 S1:v1
        vopc 02f 1111 v_cmp_tru_f16_e32 =vcc S0:v1 S1:v1
        vopc 030 1111 v_cmpx_f_f16_e32 =vcc S0:v1 S1:v1
        vopc 031 1111 v_cmpx_lt_f16_e32 =vcc S0:v1 S1:v1
        vopc 032 1111 v_cmpx_eq_f16_e32 =vcc S0:v1 S1:v1
        vopc 033 1111 v_cmpx_le_f16_e32 =vcc S0:v1 S1:v1
        vopc 034 1111 v_cmpx_gt_f16_e32 =vcc S0:v1 S1:v1
        vopc 035 1111 v_cmpx_lg_f16_e32 =vcc S0:v1 S1:v1
        vopc 036 1111 v_cmpx_ge_f16_e32 =vcc S0:v1 S1:v1
        vopc 037 1111 v_cmpx_o_f16_e32 =vcc S0:v1 S1:v1
        vopc 038 1111 v_cmpx_u_f16_e32 =vcc S0:v1 S1:v1
        vopc 039 1111 v_cmpx_nge_f16_e32 =vcc S0:v1 S1:v1
        vopc 03a 1111 v_cmpx_nlg_f16_e32 =vcc S0:v1 S1:v1
        vopc 03b 1111 v_cmpx_ngt_f16_e32 =vcc S0:v1 S1:v1
        vopc 03c 1111 v_cmpx_nle_f16_e32 =vcc S0:v1 S1:v1
        vopc 03d 1111 v_cmpx_neq_f16_e32 =vcc S0:v1 S1:v1
        vopc 03e 1111 v_cmpx_nlt_f16_e32 =vcc S0:v1 S1:v1
        vopc 03f 1111 v_cmpx_tru_f16_e32 =vcc S0:v1 S1:v1
        vopc 040 1111 v_cmp_f_f32_e32 =vcc S0:v1 S1:v1
        vopc 041 1111 v_cmp_lt_f32_e32 =vcc S0:v1 S1:v1
        vopc 042 1111 v_cmp_eq_f32_e32 =vcc S0:v1 S1:v1
        vopc 043 1111 v_cmp_le_f32_e32 =vcc S0:v1 S1:v1
        vopc 044 1111 v_cmp_gt_f32_e32 =vcc S0:v1 S1:v1
        vopc 045 1111 v_cmp_lg_f32_e32 =vcc S0:v1 S1:v1
        vopc 046 1111 v_cmp_ge_f32_e32 =vcc S0:v1 S1:v1
        vopc 047 1111 v_cmp_o_f32_e32 =vcc S0:v1 S1:v1
        vopc 048 1111 v_cmp_u_f32_e32 =vcc S0:v1 S1:v1
        vopc 049 1111 v_cmp_nge_f32_e32 =vcc S0:v1 S1:v1
        vopc 04a 1111 v_cmp_nlg_f32_e32 =vcc S0:v1 S1:v1
        vopc 04b 1111 v_cmp_ngt_f32_e32 =vcc S0:v1 S1:v1
        vopc 04c 1111 v_cmp_nle_f32_e32 =vcc S0:v1 S1:v1
        vopc 04d 1111 v_cmp_neq_f32_e32 =vcc S0:v1 S1:v1
        vopc 04e 1111 v_cmp_nlt_f32_e32 =vcc S0:v1 S1:v1
        vopc 04f 1111 v_cmp_tru_f32_e32 =vcc S0:v1 S1:v1
        vopc 050 1111 v_cmpx_f_f32_e32 =vcc S0:v1 S1:v1
        vopc 051 1111 v_cmpx_lt_f32_e32 =vcc S0:v1 S1:v1
        vopc 052 1111 v_cmpx_eq_f32_e32 =vcc S0:v1 S1:v1
        vopc 053 1111 v_cmpx_le_f32_e32 =vcc S0:v1 S1:v1
        vopc 054 1111 v_cmpx_gt_f32_e32 =vcc S0:v1 S1:v1
        vopc 055 1111 v_cmpx_lg_f32_e32 =vcc S0:v1 S1:v1
        vopc 056 1111 v_cmpx_ge_f32_e32 =vcc S0:v1 S1:v1
        vopc 057 1111 v_cmpx_o_f32_e32 =vcc S0:v1 S1:v1
        vopc 058 1111 v_cmpx_u_f32_e32 =vcc S0:v1 S1:v1
        vopc 059 1111 v_cmpx_nge_f32_e32 =vcc S0:v1 S1:v1
        vopc 05a 1111 v_cmpx_nlg_f32_e32 =vcc S0:v1 S1:v1
        vopc 05b 1111 v_cmpx_ngt_f32_e32 =vcc S0:v1 S1:v1
        vopc 05c 1111 v_cmpx_nle_f32_e32 =vcc S0:v1 S1:v1
        vopc 05d 1111 v_cmpx_neq_f32_e32 =vcc S0:v1 S1:v1
        vopc 05e 1111 v_cmpx_nlt_f32_e32 =vcc S0:v1 S1:v1
        vopc 05f 1111 v_cmpx_tru_f32_e32 =vcc S0:v1 S1:v1
        vopc 060 1111 v_cmp_f_f64_e32 =vcc S0:v2 S1:v2
        vopc 061 1111 v_cmp_lt_f64_e32 =vcc S0:v2 S1:v2
        vopc 062 1111 v_cmp_eq_f64_e32 =vcc S0:v2 S1:v2
        vopc 063 1111 v_cmp_le_f64_e32 =vcc S0:v2 S1:v2
        vopc 064 1111 v_cmp_gt_f64_e32 =vcc S0:v2 S1:v2
        vopc 065 1111 v_cmp_lg_f64_e32 =vcc S0:v2 S1:v2
        vopc 066 1111 v_cmp_ge_f64_e32 =vcc S0:v2 S1:v2
        vopc 067 1111 v_cmp_o_f64_e32 =vcc S0:v2 S1:v2
        vopc 068 1111 v_cmp_u_f64_e32 =vcc S0:v2 S1:v2
        vopc 069 1111 v_cmp_nge_f64_e32 =vcc S0:v2 S1:v2
        vopc 06a 1111 v_cmp_nlg_f64_e32 =vcc S0:v2 S1:v2
        vopc 06b 1111 v_cmp_ngt_f64_e32 =vcc S0:v2 S1:v2
        vopc 06c 1111 v_cmp_nle_f64_e32 =vcc S0:v2 S1:v2
        vopc 06d 1111 v_cmp_neq_f64_e32 =vcc S0:v2 S1:v2
        vopc 06e 1111 v_cmp_nlt_f64_e32 =vcc S0:v2 S1:v2
        vopc 06f 1111 v_cmp_tru_f64_e32 =vcc S0:v2 S1:v2
        vopc 070 1111 v_cmpx_f_f64_e32 =vcc S0:v2 S1:v2
        vopc 071 1111 v_cmpx_lt_f64_e32 =vcc S0:v2 S1:v2
        vopc 072 1111 v_cmpx_eq_f64_e32 =vcc S0:v2 S1:v2
        vopc 073 1111 v_cmpx_le_f64_e32 =vcc S0:v2 S1:v2
        vopc 074 1111 v_cmpx_gt_f64_e32 =vcc S0:v2 S1:v2
        vopc 075 1111 v_cmpx_lg_f64_e32 =vcc S0:v2 S1:v2
        vopc 076 1111 v_cmpx_ge_f64_e32 =vcc S0:v2 S1:v2
        vopc 077 1111 v_cmpx_o_f64_e32 =vcc S0:v2 S1:v2
        vopc 078 1111 v_cmpx_u_f64_e32 =vcc S0:v2 S1:v2
        vopc 079 1111 v_cmpx_nge_f64_e32 =vcc S0:v2 S1:v2
        vopc 07a 1111 v_cmpx_nlg_f64_e32 =vcc S0:v2 S1:v2
        vopc 07b 1111 v_cmpx_ngt_f64_e32 =vcc S0:v2 S1:v2
        vopc 07c 1111 v_cmpx_nle_f64_e32 =vcc S0:v2 S1:v2
        vopc 07d 1111 v_cmpx_neq_f64_e32 =vcc S0:v2 S1:v2
        vopc 07e 1111 v_cmpx_nlt_f64_e32 =vcc S0:v2 S1:v2
        vopc 07f 1111 v_cmpx_tru_f64_e32 =vcc S0:v2 S1:v2
        vopc 0a0 1111 v_cmp_f_i16_e32 =vcc S0:v1 S1:v1
        vopc 0a1 1111 v_cmp_lt_i16_e32 =vcc S0:v1 S1:v1
        vopc 0a2 1111 v_cmp_eq_i16_e32 =vcc S0:v1 S1:v1
        vopc 0a3 1111 v_cmp_le_i16_e32 =vcc S0:v1 S1:v1
        vopc 0a4 1111 v_cmp_gt_i16_e32 =vcc S0:v1 S1:v1
        vopc 0a5 1111 v_cmp_ne_i16_e32 =vcc S0:v1 S1:v1
        vopc 0a6 1111 v_cmp_ge_i16_e32 =vcc S0:v1 S1:v1
        vopc 0a7 1111 v_cmp_t_i16_e32 =vcc S0:v1 S1:v1
        vopc 0a8 1111 v_cmp_f_u16_e32 =vcc S0:v1 S1:v1
        vopc 0a9 1111 v_cmp_lt_u16_e32 =vcc S0:v1 S1:v1
        vopc 0aa 1111 v_cmp_eq_u16_e32 =vcc S0:v1 S1:v1
        vopc 0ab 1111 v_cmp_le_u16_e32 =vcc S0:v1 S1:v1
        vopc 0ac 1111 v_cmp_gt_u16_e32 =vcc S0:v1 S1:v1
        vopc 0ad 1111 v_cmp_ne_u16_e32 =vcc S0:v1 S1:v1
        vopc 0ae 1111 v_cmp_ge_u16_e32 =vcc S0:v1 S1:v1
        vopc 0af 1111 v_cmp_t_u16_e32 =vcc S0:v1 S1:v1
        vopc 0b0 1111 v_cmpx_f_i16_e32 =vcc S0:v1 S1:v1
        vopc 0b1 1111 v_cmpx_lt_i16_e32 =vcc S0:v1 S1:v1
        vopc 0b2 1111 v_cmpx_eq_i16_e32 =vcc S0:v1 S1:v1
        vopc 0b3 1111 v_cmpx_le_i16_e32 =vcc S0:v1 S1:v1
        vopc 0b4 1111 v_cmpx_gt_i16_e32 =vcc S0:v1 S1:v1
        vopc 0b5 1111 v_cmpx_ne_i16_e32 =vcc S0:v1 S1:v1
        vopc 0b6 1111 v_cmpx_ge_i16_e32 =vcc S0:v1 S1:v1
        vopc 0b7 1111 v_cmpx_t_i16_e32 =vcc S0:v1 S1:v1
        vopc 0b8 1111 v_cmpx_f_u16_e32 =vcc S0:v1 S1:v1
        vopc 0b9 1111 v_cmpx_lt_u16_e32 =vcc S0:v1 S1:v1
        vopc 0ba 1111 v_cmpx_eq_u16_e32 =vcc S0:v1 S1:v1
        vopc 0bb 1111 v_cmpx_le_u16_e32 =vcc S0:v1 S1:v1
        vopc 0bc 1111 v_cmpx_gt_u16_e32 =vcc S0:v1 S1:v1
        vopc 0bd 1111 v_cmpx_ne_u16_e32 =vcc S0:v1 S1:v1
        vopc 0be 1111 v_cmpx_ge_u16_e32 =vcc S0:v1 S1:v1
        vopc 0bf 1111 v_cmpx_t_u16_e32 =vcc S0:v1 S1:v1
        vopc 0c0 1111 v_cmp_f_i32_e32 =vcc S0:v1 S1:v1
        vopc 0c1 1111 v_cmp_lt_i32_e32 =vcc S0:v1 S1:v1
        vopc 0c2 1111 v_cmp_eq_i32_e32 =vcc S0:v1 S1:v1
        vopc 0c3 1111 v_cmp_le_i32_e32 =vcc S0:v1 S1:v1
        vopc 0c4 1111 v_cmp_gt_i32_e32 =vcc S0:v1 S1:v1
        vopc 0c5 1111 v_cmp_ne_i32_e32 =vcc S0:v1 S1:v1
        vopc 0c6 1111 v_cmp_ge_i32_e32 =vcc S0:v1 S1:v1
        vopc 0c7 1111 v_cmp_t_i32_e32 =vcc S0:v1 S1:v1
        vopc 0c8 1111 v_cmp_f_u32_e32 =vcc S0:v1 S1:v1
        vopc 0c9 1111 v_cmp_lt_u32_e32 =vcc S0:v1 S1:v1
        vopc 0ca 1111 v_cmp_eq_u32_e32 =vcc S0:v1 S1:v1
        vopc 0cb 1111 v_cmp_le_u32_e32 =vcc S0:v1 S1:v1
        vopc 0cc 1111 v_cmp_gt_u32_e32 =vcc S0:v1 S1:v1
        vopc 0cd 1111 v_cmp_ne_u32_e32 =vcc S0:v1 S1:v1
        vopc 0ce 1111 v_cmp_ge_u32_e32 =vcc S0:v1 S1:v1
        vopc 0cf 1111 v_cmp_t_u32_e32 =vcc S0:v1 S1:v1
        vopc 0d0 1111 v_cmpx_f_i32_e32 =vcc S0:v1 S1:v1
        vopc 0d1 1111 v_cmpx_lt_i32_e32 =vcc S0:v1 S1:v1
        vopc 0d2 1111 v_cmpx_eq_i32_e32 =vcc S0:v1 S1:v1
        vopc 0d3 1111 v_cmpx_le_i32_e32 =vcc S0:v1 S1:v1
        vopc 0d4 1111 v_cmpx_gt_i32_e32 =vcc S0:v1 S1:v1
        vopc 0d5 1111 v_cmpx_ne_i32_e32 =vcc S0:v1 S1:v1
        vopc 0d6 1111 v_cmpx_ge_i32_e32 =vcc S0:v1 S1:v1
        vopc 0d7 1111 v_cmpx_t_i32_e32 =vcc S0:v1 S1:v1
        vopc 0d8 1111 v_cmpx_f_u32_e32 =vcc S0:v1 S1:v1
        vopc 0d9 1111 v_cmpx_lt_u32_e32 =vcc S0:v1 S1:v1
        vopc 0da 1111 v_cmpx_eq_u32_e32 =vcc S0:v1 S1:v1
        vopc 0db 1111 v_cmpx_le_u32_e32 =vcc S0:v1 S1:v1
        vopc 0dc 1111 v_cmpx_gt_u32_e32 =vcc S0:v1 S1:v1
        vopc 0dd 1111 v_cmpx_ne_u32_e32 =vcc S0:v1 S1:v1
        vopc 0de 1111 v_cmpx_ge_u32_e32 =vcc S0:v1 S1:v1
        vopc 0df 1111 v_cmpx_t_u32_e32 =vcc S0:v1 S1:v1
        vopc 0e0 1111 v_cmp_f_i64_e32 =vcc S0:v2 S1:v2
        vopc 0e1 1111 v_cmp_lt_i64_e32 =vcc S0:v2 S1:v2
        vopc 0e2 1111 v_cmp_eq_i64_e32 =vcc S0:v2 S1:v2
        vopc 0e3 1111 v_cmp_le_i64_e32 =vcc S0:v2 S1:v2
        vopc 0e4 1111 v_cmp_gt_i64_e32 =vcc S0:v2 S1:v2
        vopc 0e5 1111 v_cmp_ne_i64_e32 =vcc S0:v2 S1:v2
        vopc 0e6 1111 v_cmp_ge_i64_e32 =vcc S0:v2 S1:v2
        vopc 0e7 1111 v_cmp_t_i64_e32 =vcc S0:v2 S1:v2
        vopc 0e8 1111 v_cmp_f_u64_e32 =vcc S0:v2 S1:v2
        vopc 0e9 1111 v_cmp_lt_u64_e32 =vcc S0:v2 S1:v2
        vopc 0ea 1111 v_cmp_eq_u64_e32 =vcc S0:v2 S1:v2
        vopc 0eb 1111 v_cmp_le_u64_e32 =vcc S0:v2 S1:v2
        vopc 0ec 1111 v_cmp_gt_u64_e32 =vcc S0:v2 S1:v2
        vopc 0ed 1111 v_cmp_ne_u64_e32 =vcc S0:v2 S1:v2
        vopc 0ee 1111 v_cmp_ge_u64_e32 =vcc S0:v2 S1:v2
        vopc 0ef 1111 v_cmp_t_u64_e32 =vcc S0:v2 S1:v2
        vopc 0f0 1111 v_cmpx_f_i64_e32 =vcc S0:v2 S1:v2
        vopc 0f1 1111 v_cmpx_lt_i64_e32 =vcc S0:v2 S1:v2
        vopc 0f2 1111 v_cmpx_eq_i64_e32 =vcc S0:v2 S1:v2
        vopc 0f3 1111 v_cmpx_le_i64_e32 =vcc S0:v2 S1:v2
        vopc 0f4 1111 v_cmpx_gt_i64_e32 =vcc S0:v2 S1:v2
        vopc 0f5 1111 v_cmpx_ne_i64_e32 =vcc S0:v2 S1:v2
        vopc 0f6 1111 v_cmpx_ge_i64_e32 =vcc S0:v2 S1:v2
        vopc 0f7 1111 v_cmpx_t_i64_e32 =vcc S0:v2 S1:v2
        vopc 0f8 1111 v_cmpx_f_u64_e32 =vcc S0:v2 S1:v2
        vopc 0f9 1111 v_cmpx_lt_u64_e32 =vcc S0:v2 S1:v2
        vopc 0fa 1111 v_cmpx_eq_u64_e32 =vcc S0:v2 S1:v2
        vopc 0fb 1111 v_cmpx_le_u64_e32 =vcc S0:v2 S1:v2
        vopc 0fc 1111 v_cmpx_gt_u64_e32 =vcc S0:v2 S1:v2
        vopc 0fd 1111 v_cmpx_ne_u64_e32 =vcc S0:v2 S1:v2
        vopc 0fe 1111 v_cmpx_ge_u64_e32 =vcc S0:v2 S1:v2
        vopc 0ff 1111 v_cmpx_t_u64_e32 =vcc S0:v2 S1:v2
        vop3 010 1111 v_cmp_class_f32_e64 D:s2 S0:v1 S1:v1
        vop3 011 1111 v_cmpx_class_f32_e64 D:s2 S0:v1 S1:v1
        vop3 012 1111 v_cmp_class_f64_e64 D:s2 S0:v2 S1:v1
        vop3 013 1111 v_cmpx_class_f64_e64 D:s2 S0:v2 S1:v1
        vop3 014 1111 v_cmp_class_f16_e64 D:s2 S0:v1 S1:v1
        vop3 015 1111 v_cmpx_class_f16_e64 D:s2 S0:v1 S1:v1
        vop3 020 1111 v_cmp_f_f16_e64 D:s2 S0:v1 S1:v1
        vop3 021 1111 v_cmp_lt_f16_e64 D:s2 S0:v1 S1:v1
        vop3 022 1111 v_cmp_eq_f16_e64 D:s2 S0:v1 S1:v1
        vop3 023 1111 v_cmp_le_f16_e64 D:s2 S0:v1 S1:v1
        vop3 024 1111 v_cmp_gt_f16_e64 D:s2 S0:v1 S1:v1
        vop3 025 1111 v_cmp_lg_f16_e64 D:s2 S0:v1 S1:v1
        vop3 026 1111 v_cmp_ge_f16_e64 D:s2 S0:v1 S1:v1
        vop3 027 1111 v_cmp_o_f16_e64 D:s2 S0:v1 S1:v1
        vop3 028 1111 v_cmp_u_f16_e64 D:s2 S0:v1 S1:v1
        vop3 029 1111 v_cmp_nge_f16_e64 D:s2 S0:v1 S1:v1
        vop3 02a 1111 v_cmp_nlg_f16_e64 D:s2 S0:v1 S1:v1
        vop3 02b 1111 v_cmp_ngt_f16_e64 D:s2 S0:v1 S1:v1
        vop3 02c 1111 v_cmp_nle_f16_e64 D:s2 S0:v1 S1:v1
        vop3 02d 1111 v_cmp_neq_f16_e64 D:s2 S0:v1 S1:v1
        vop3 02e 1111 v_cmp_nlt_f16_e64 D:s2 S0:v1 S1:v1
        vop3 02f 1111 v_cmp_tru_f16_e64 D:s2 S0:v1 S1:v1
        vop3 030 1111 v_cmpx_f_f16_e64 D:s2 S0:v1 S1:v1
        vop3 031 1111 v_cmpx_lt_f16_e64 D:s2 S0:v1 S1:v1
        vop3 032 1111 v_cmpx_eq_f16_e64 D:s2 S0:v1 S1:v1
        vop3 033 1111 v_cmpx_le_f16_e64 D:s2 S0:v1 S1:v1
        vop3 034 1111 v_cmpx_gt_f16_e64 D:s2 S0:v1 S1:v1
        vop3 035 1111 v_cmpx_lg_f16_e64 D:s2 S0:v1 S1:v1
        vop3 036 1111 v_cmpx_ge_f16_e64 D:s2 S0:v1 S1:v1
        vop3 037 1111 v_cmpx_o_f16_e64 D:s2 S0:v1 S1:v1
        vop3 038 1111 v_cmpx_u_f16_e64 D:s2 S0:v1 S1:v1
        vop3 039 1111 v_cmpx_nge_f16_e64 D:s2 S0:v1 S1:v1
        vop3 03a 1111 v_cmpx_nlg_f16_e64 D:s2 S0:v1 S1:v1
        vop3 03b 1111 v_cmpx_ngt_f16_e64 D:s2 S0:v1 S1:v1
        vop3 03c 1111 v_cmpx_nle_f16_e64 D:s2 S0:v1 S1:v1
        vop3 03d 1111 v_cmpx_neq_f16_e64 D:s2 S0:v1 S1:v1
        vop3 03e 1111 v_cmpx_nlt_f16_e64 D:s2 S0:v1 S1:v1
        vop3 03f 1111 v_cmpx_tru_f16_e64 D:s2 S0:v1 S1:v1
        vop3 040 1111 v_cmp_f_f32_e64 D:s2 S0:v1 S1:v1
        vop3 041 1111 v_cmp_lt_f32_e64 D:s2 S0:v1 S1:v1
        vop3 042 1111 v_cmp_eq_f32_e64 D:s2 S0:v1 S1:v1
        vop3 043 1111 v_cmp_le_f32_e64 D:s2 S0:v1 S1:v1
        vop3 044 1111 v_cmp_gt_f32_e64 D:s2 S0:v1 S1:v1
        vop3 045 1111 v_cmp_lg_f32_e64 D:s2 S0:v1 S1:v1
        vop3 046 1111 v_cmp_ge_f32_e64 D:s2 S0:v1 S1:v1
        vop3 047 1111 v_cmp_o_f32_e64 D:s2 S0:v1 S1:v1
        vop3 048 1111 v_cmp_u_f32_e64 D:s2 S0:v1 S1:v1
        vop3 049 1111 v_cmp_nge_f32_e64 D:s2 S0:v1 S1:v1
        vop3 04a 1111 v_cmp_nlg_f32_e64 D:s2 S0:v1 S1:v1
        vop3 04b 1111 v_cmp_ngt_f32_e64 D:s2 S0:v1 S1:v1
        vop3 04c 1111 v_cmp_nle_f32_e64 D:s2 S0:v1 S1:v1
        vop3 04d 1111 v_cmp_neq_f32_e64 D:s2 S0:v1 S1:v1
        vop3 04e 1111 v_cmp_nlt_f32_e64 D:s2 S0:v1 S1:v1
        vop3 04f 1111 v_cmp_tru_f32_e64 D:s2 S0:v1 S1:v1
        vop3 050 1111 v_cmpx_f_f32_e64 D:s2 S0:v1 S1:v1
        vop3 051 1111 v_cmpx_lt_f32_e64 D:s2 S0:v1 S1:v1
        vop3 052 1111 v_cmpx_eq_f32_e64 D:s2 S0:v1 S1:v1
        vop3 053 1111 v_cmpx_le_f32_e64 D:s2 S0:v1 S1:v1
        vop3 054 1111 v_cmpx_gt_f32_e64 D:s2 S0:v1 S1:v1
        vop3 055 1111 v_cmpx_lg_f32_e64 D:s2 S0:v1 S1:v1
        vop3 056 1111 v_cmpx_ge_f32_e64 D:s2 S0:v1 S1:v1
        vop3 057 1111 v_cmpx_o_f32_e64 D:s2 S0:v1 S1:v1
        vop3 058 1111 v_cmpx_u_f32_e64 D:s2 S0:v1 S1:v1
        vop3 059 1111 v_cmpx_nge_f32_e64 D:s2 S0:v1 S1:v1
        vop3 05a 1111 v_cmpx_nlg_f32_e64 D:s2 S0:v1 S1:v1
        vop3 05b 1111 v_cmpx_ngt_f32_e64 D:s2 S0:v1 S1:v1
        vop3 05c 1111 v_cmpx_nle_f32_e64 D:s2 S0:v1 S1:v1
        vop3 05d 1111 v_cmpx_neq_f32_e64 D:s2 S0:v1 S1:v1
        vop3 05e 1111 v_cmpx_nlt_f32_e64 D:s2 S0:v1 S1:v1
        vop3 05f 1111 v_cmpx_tru_f32_e64 D:s2 S0:v1 S1:v1
        vop3 060 1111 v_cmp_f_f64_e64 D:s2 S0:v2 S1:v2
        vop3 061 1111 v_cmp_lt_f64_e64 D:s2 S0:v2 S1:v2
        vop3 062 1111 v_cmp_eq_f64_e64 D:s2 S0:v2 S1:v2
        vop3 063 1111 v_cmp_le_f64_e64 D:s2 S0:v2 S1:v2
        vop3 064 1111 v_cmp_gt_f64_e64 D:s2 S0:v2 S1:v2
        vop3 065 1111 v_cmp_lg_f64_e64 D:s2 S0:v2 S1:v2
        vop3 066 1111 v_cmp_ge_f64_e64 D:s2 S0:v2 S1:v2
        vop3 067 1111 v_cmp_o_f64_e64 D:s2 S0:v2 S1:v2
        vop3 068 1111 v_cmp_u_f64_e64 D:s2 S0:v2 S1:v2
        vop3 069 1111 v_cmp_nge_f64_e64 D:s2 S0:v2 S1:v2
        vop3 06a 1111 v_cmp_nlg_f64_e64 D:s2 S0:v2 S1:v2
        vop3 06b 1111 v_cmp_ngt_f64_e64 D:s2 S0:v2 S1:v2
        vop3 06c 1111 v_cmp_nle_f64_e64 D:s2 S0:v2 S1:v2
        vop3 06d 1111 v_cmp_neq_f64_e64 D:s2 S0:v2 S1:v2
        vop3 06e 1111 v_cmp_nlt_f64_e64 D:s2 S0:v2 S1:v2
        vop3 06f 1111 v_cmp_tru_f64_e64 D:s2 S0:v2 S1:v2
        vop3 070 1111 v_cmpx_f_f64_e64 D:s2 S0:v2 S1:v2
        vop3 071 1111 v_cmpx_lt_f64_e64 D:s2 S0:v2 S1:v2
        vop3 072 1111 v_cmpx_eq_f64_e64 D:s2 S0:v2 S1:v2
        vop3 073 1111 v_cmpx_le_f64_e64 D:s2 S0:v2 S1:v2
        vop3 074 1111 v_cmpx_gt_f64_e64 D:s2 S0:v2 S1:v2
        vop3 075 1111 v_cmpx_lg_f64_e64 D:s2 S0:v2 S1:v2
        vop3 076 1111 v_cmpx_ge_f64_e64 D:s2 S0:v2 S1:v2
        vop3 077 1111 v_cmpx_o_f64_e64 D:s2 S0:v2 S1:v2
        vop3 078 1111 v_cmpx_u_f64_e64 D:s2 S0:v2 S1:v2
        vop3 079 1111 v_cmpx_nge_f64_e64 D:s2 S0:v2 S1:v2
        vop3 07a 1111 v_cmpx_nlg_f64_e64 D:s2 S0:v2 S1:v2
        vop3 07b 1111 v_cmpx_ngt_f64_e64 D:s2 S0:v2 S1:v2
        vop3 07c 1111 v_cmpx_nle_f64_e64 D:s2 S0:v2 S1:v2
        vop3 07d 1111 v_cmpx_neq_f64_e64 D:s2 S0:v2 S1:v2
        vop3 07e 1111 v_cmpx_nlt_f64_e64 D:s2 S0:v2 S1:v2
        vop3 07f 1111 v_cmpx_tru_f64_e64 D:s2 S0:v2 S1:v2
        vop3 0a0 1111 v_cmp_f_i16_e64 D:s2 S0:v1 S1:v1
        vop3 0a1 1111 v_cmp_lt_i16_e64 D:s2 S0:v1 S1:v1
        vop3 0a2 1111 v_cmp_eq_i16_e64 D:s2 S0:v1 S1:v1
        vop3 0a3 1111 v_cmp_le_i16_e64 D:s2 S0:v1 S1:v1
        vop3 0a4 1111 v_cmp_gt_i16_e64 D:s2 S0:v1 S1:v1
        vop3 0a5 1111 v_cmp_ne_i16_e64 D:s2 S0:v1 S1:v1
        vop3 0a6 1111 v_cmp_ge_i16_e64 D:s2 S0:v1 S1:v1
        vop3 0a7 1111 v_cmp_t_i16_e64 D:s2 S0:v1 S1:v1
        vop3 0a8 1111 v_cmp_f_u16_e64 D:s2 S0:v1 S1:v1
        vop3 0a9 1111 v_cmp_lt_u16_e64 D:s2 S0:v1 S1:v1
        vop3 0aa 1111 v_cmp_eq_u16_e64 D:s2 S0:v1 S1:v1
        vop3 0ab 1111 v_cmp_le_u16_e64 D:s2 S0:v1 S1:v1
        vop3 0ac 1111 v_cmp_gt_u16_e64 D:s2 S0:v1 S1:v1
        vop3 0ad 1111 v_cmp_ne_u16_e64 D:s2 S0:v1 S1:v1
        vop3 0ae 1111 v_cmp_ge_u16_e64 D:s2 S0:v1 S1:v1
        vop3 0af 1111 v_cmp_t_u16_e64 D:s2 S0:v1 S1:v1
        vop3 0b0 1111 v_cmpx_f_i16_e64 D:s2 S0:v1 S1:v1
        vop3 0b1 1111 v_cmpx_lt_i16_e64 D:s2 S0:v1 S1:v1
        vop3 0b2 1111 v_cmpx_eq_i16_e64 D:s2 S0:v1 S1:v1
        vop3 0b3 1111 v_cmpx_le_i16_e64 D:s2 S0:v1 S1:v1
        vop3 0b4 1111 v_cmpx_gt_i16_e64 D:s2 S0:v1 S1:v1
        vop3 0b5 1111 v_cmpx_ne_i16_e64 D:s2 S0:v1 S1:v1
        vop3 0b6 1111 v_cmpx_ge_i16_e64 D:s2 S0:v1 S1:v1
        vop3 0b7 1111 v_cmpx_t_i16_e64 D:s2 S0:v1 S1:v1
        vop3 0b8 1111 v_cmpx_f_u16_e64 D:s2 S0:v1 S1:v1
        vop3 0b9 1111 v_cmpx_lt_u16_e64 D:s2 S0:v1 S1:v1
        vop3 0ba 1111 v_cmpx_eq_u16_e64 D:s2 S0:v1 S1:v1
        vop3 0bb 1111 v_cmpx_le_u16_e64 D:s2 S0:v1 S1:v1
        vop3 0bc 1111 v_cmpx_gt_u16_e64 D:s2 S0:v1 S1:v1
        vop3 0bd 1111 v_cmpx_ne_u16_e64 D:s2 S0:v1 S1:v1
        vop3 0be 1111 v_cmpx_ge_u16_e64 D:s2 S0:v1 S1:v1
        vop3 0bf 1111 v_cmpx_t_u16_e64 D:s2 S0:v1 S1:v1
        vop3 0c0 1111 v_cmp_f_i32_e64 D:s2 S0:v1 S1:v1
        vop3 0c1 1111 v_cmp_lt_i32_e64 D:s2 S0:v1 S1:v1
        vop3 0c2 1111 v_cmp_eq_i32_e64 D:s2 S0:v1 S1:v1
        vop3 0c3 1111 v_cmp_le_i32_e64 D:s2 S0:v1 S1:v1
        vop3 0c4 1111 v_cmp_gt_i32_e64 D:s2 S0:v1 S1:v1
        vop3 0c5 1111 v_cmp_ne_i32_e64 D:s2 S0:v1 S1:v1
        vop3 0c6 1111 v_cmp_ge_i32_e64 D:s2 S0:v1 S1:v1
        vop3 0c7 1111 v_cmp_t_i32_e64 D:s2 S0:v1 S1:v1
        vop3 0c8 1111 v_cmp_f_u32_e64 D:s2 S0:v1 S1:v1
        vop3 0c9 1111 v_cmp_lt_u32_e64 D:s2 S0:v1 S1:v1
        vop3 0ca 1111 v_cmp_eq_u32_e64 D:s2 S0:v1 S1:v1
        vop3 0cb 1111 v_cmp_le_u32_e64 D:s2 S0:v1 S1:v1
        vop3 0cc 1111 v_cmp_gt_u32_e64 D:s2 S0:v1 S1:v1
        vop3 0cd 1111 v_cmp_ne_u32_e64 D:s2 S0:v1 S1:v1
        vop3 0ce 1111 v_cmp_ge_u32_e64 D:s2 S0:v1 S1:v1
        vop3 0cf 1111 v_cmp_t_u32_e64 D:s2 S0:v1 S1:v1
        vop3 0d0 1111 v_cmpx_f_i32_e64 D:s2 S0:v1 S1:v1
        vop3 0d1 1111 v_cmpx_lt_i32_e64 D:s2 S0:v1 S1:v1
        vop3 0d2 1111 v_cmpx_eq_i32_e64 D:s2 S0:v1 S1:v1
        vop3 0d3 1111 v_cmpx_le_i32_e64 D:s2 S0:v1 S1:v1
        vop3 0d4 1111 v_cmpx_gt_i32_e64 D:s2 S0:v1 S1:v1
        vop3 0d5 1111 v_cmpx_ne_i32_e64 D:s2 S0:v1 S1:v1
        vop3 0d6 1111 v_cmpx_ge_i32_e64 D:s2 S0:v1 S1:v1
        vop3 0d7 1111 v_cmpx_t_i32_e64 D:s2 S0:v1 S1:v1
        vop3 0d8 1111 v_cmpx_f_u32_e64 D:s2 S0:v1 S1:v1
        vop3 0d9 1111 v_cmpx_lt_u32_e64 D:s2 S0:v1 S1:v1
        vop3 0da 1111 v_cmpx_eq_u32_e64 D:s2 S0:v1 S1:v1
        vop3 0db 1111 v_cmpx_le_u32_e64 D:s2 S0:v1 S1:v1
        vop3 0dc 1111 v_cmpx_gt_u32_e64 D:s2 S0:v1 S1:v1
        vop3 0dd 1111 v_cmpx_ne_u32_e64 D:s2 S0:v1 S1:v1
        vop3 0de 1111 v_cmpx_ge_u32_e64 D:s2 S0:v1 S1:v1
        vop3 0df 1111 v_cmpx_t_u32_e64 D:s2 S0:v1 S1:v1
        vop3 0e0 1111 v_cmp_f_i64_e64 D:s2 S0:v2 S1:v2
        vop3 0e1 1111 v_cmp_lt_i64_e64 D:s2 S0:v2 S1:v2
        vop3 0e2 1111 v_cmp_eq_i64_e64 D:s2 S0:v2 S1:v2
        vop3 0e3 1111 v_cmp_le_i64_e64 D:s2 S0:v2 S1:v2
        vop3 0e4 1111 v_cmp_gt_i64_e64 D:s2 S0:v2 S1:v2
        vop3 0e5 1111 v_cmp_ne_i64_e64 D:s2 S0:v2 S1:v2
        vop3 0e6 1111 v_cmp_ge_i64_e64 D:s2 S0:v2 S1:v2
        vop3 0e7 1111 v_cmp_t_i64_e64 D:s2 S0:v2 S1:v2
        vop3 0e8 1111 v_cmp_f_u64_e64 D:s2 S0:v2 S1:v2
        vop3 0e9 1111 v_cmp_lt_u64_e64 D:s2 S0:v2 S1:v2
        vop3 0ea 1111 v_cmp_eq_u64_e64 D:s2 S0:v2 S1:v2
        vop3 0eb 1111 v_cmp_le_u64_e64 D:s2 S0:v2 S1:v2
        vop3 0ec 1111 v_cmp_gt_u64_e64 D:s2 S0:v2 S1:v2
        vop3 0ed 1111 v_cmp_ne_u64_e64 D:s2 S0:v2 S1:v2
        vop3 0ee 1111 v_cmp_ge_u64_e64 D:s2 S0:v2 S1:v2
        vop3 0ef 1111 v_cmp_t_u64_e64 D:s2 S0:v2 S1:v2
        vop3 0f0 1111 v_cmpx_f_i64_e64 D:s2 S0:v2 S1:v2
        vop3 0f1 1111 v_cmpx_lt_i64_e64 D:s2 S0:v2 S1:v2
        vop3 0f2 1111 v_cmpx_eq_i64_e64 D:s2 S0:v2 S1:v2
        vop3 0f3 1111 v_cmpx_le_i64_e64 D:s2 S0:v2 S1:v2
        vop3 0f4 1111 v_cmpx_gt_i64_e64 D:s2 S0:v2 S1:v2
        vop3 0f5 1111 v_cmpx_ne_i64_e64 D:s2 S0:v2 S1:v2
        vop3 0f6 1111 v_cmpx_ge_i64_e64 D:s2 S0:v2 S1:v2
        vop3 0f7 1111 v_cmpx_t_i64_e64 D:s2 S0:v2 S1:v2
        vop3 0f8 1111 v_cmpx_f_u64_e64 D:s2 S0:v2 S1:v2
        vop3 0f9 1111 v_cmpx_lt_u64_e64 D:s2 S0:v2 S1:v2
        vop3 0fa 1111 v_cmpx_eq_u64_e64 D:s2 S0:v2 S1:v2
        vop3 0fb 1111 v_cmpx_le_u64_e64 D:s2 S0:v2 S1:v2
        vop3 0fc 1111 v_cmpx_gt_u64_e64 D:s2 S0:v2 S1:v2
        vop3 0fd 1111 v_cmpx_ne_u64_e64 D:s2 S0:v2 S1:v2
        vop3 0fe 1111 v_cmpx_ge_u64_e64 D:s2 S0:v2 S1:v2
        vop3 0ff 1111 v_cmpx_t_u64_e64 D:s2 S0:v2 S1:v2
        vop3 100 1111 v_cndmask_b32_e64 D:v1 S0:v1 S1:v1 S2:s2
        vop3 101 1111 v_add_f32_e64 D:v1 S0:v1 S1:v1
        vop3 102 1111 v_sub_f32_e64 D:v1 S0:v1 S1:v1
        vop3 103 1111 v_subrev_f32_e64 D:v1 S0:v1 S1:v1
        vop3 104 1111 v_mul_legacy_f32_e64 D:v1 S0:v1 S1:v1
        vop3 105 1111 v_mul_f32_e64 D:v1 S0:v1 S1:v1
        vop3 106 1111 v_mul_i32_i24_e64 D:v1 S0:v1 S1:v1
        vop3 107 1111 v_mul_hi_i32_i24_e64 D:v1 S0:v1 S1:v1
        vop3 108 1111 v_mul_u32_u24_e64 D:v1 S0:v1 S1:v1
        vop3 109 1111 v_mul_hi_u32_u24_e64 D:v1 S0:v1 S1:v1
        vop3 10a 1111 v_min_f32_e64 D:v1 S0:v1 S1:v1
        vop3 10b 1111 v_max_f32_e64 D:v1 S0:v1 S1:v1
        vop3 10c 1111 v_min_i32_e64 D:v1 S0:v1 S1:v1
        vop3 10d 1111 v_max_i32_e64 D:v1 S0:v1 S1:v1
        vop3 10e 1111 v_min_u32_e64 D:v1 S0:v1 S1:v1
        vop3 10f 1111 v_max_u32_e64 D:v1 S0:v1 S1:v1
        vop3 110 1111 v_lshrrev_b32_e64 D:v1 S0:v1 S1:v1
        vop3 111 1111 v_ashrrev_i32_e64 D:v1 S0:v1 S1:v1
        vop3 112 1111 v_lshlrev_b32_e64 D:v1 S0:v1 S1:v1
        vop3 113 1111 v_and_b32_e64 D:v1 S0:v1 S1:v1
        vop3 114 1111 v_or_b32_e64 D:v1 S0:v1 S1:v1
        vop3 115 1111 v_xor_b32_e64 D:v1 S0:v1 S1:v1
        vop3 116 1111 v_mac_f32_e64 D:v1 S0:v1 S1:v1
        vop3 119 1111 v_add_co_u32_e64 D:v1 SD:s2 S0:v1 S1:v1
        vop3 11a 1111 v_sub_co_u32_e64 D:v1 SD:s2 S0:v1 S1:v1
        vop3 11b 1111 v_subrev_co_u32_e64 D:v1 SD:s2 S0:v1 S1:v1
        vop3 11c 1111 v_addc_co_u32_e64 D:v1 SD:s2 S0:v1 S1:v1 S2:s2
        vop3 11d 1111 v_subb_co_u32_e64 D:v1 SD:s2 S0:v1 S1:v1 S2:s2
        vop3 11e 1111 v_subbrev_co_u32_e64 D:v1 SD:s2 S0:v1 S1:v1 S2:s2
        vop3 11f 1111 v_add_f16_e64 D:v1 S0:v1 S1:v1
        vop3 120 1111 v_sub_f16_e64 D:v1 S0:v1 S1:v1
        vop3 121 1111 v_subrev_f16_e64 D:v1 S0:v1 S1:v1
        vop3 122 1111 v_mul_f16_e64 D:v1 S0:v1 S1:v1
        vop3 123 1111 v_mac_f16_e64 D:v1 S0:v1 S1:v1
        vop3 126 1111 v_add_u16_e64 D:v1 S0:v1 S1:v1
        vop3 127 1111 v_sub_u16_e64 D:v1 S0:v1 S1:v1
        vop3 128 1111 v_subrev_u16_e64 D:v1 S0:v1 S1:v1
        vop3 129 1111 v_mul_lo_u16_e64 D:v1 S0:v1 S1:v1
        vop3 12a 1111 v_lshlrev_b16_e64 D:v1 S0:v1 S1:v1
        vop3 12b 1111 v_lshrrev_b16_e64 D:v1 S0:v1 S1:v1
        vop3 12c 1111 v_ashrrev_i16_e64 D:v1 S0:v1 S1:v1
        vop3 12d 1111 v_max_f16_e64 D:v1 S0:v1 S1:v1
        vop3 12e 1111 v_min_f16_e64 D:v1 S0:v1 S1:v1
        vop3 12f 1111 v_max_u16_e64 D:v1 S0:v1 S1:v1
        vop3 130 1111 v_max_i16_e64 D:v1 S0:v1 S1:v1
        vop3 131 1111 v_min_u16_e64 D:v1 S0:v1 S1:v1
        vop3 132 1111 v_min_i16_e64 D:v1 S0:v1 S1:v1
        vop3 133 1111 v_ldexp_f16_e64 D:v1 S0:v1 S1:v1
        vop3 134 1111 v_add_u32_e64 D:v1 S0:v1 S1:v1
        vop3 135 1111 v_sub_u32_e64 D:v1 S0:v1 S1:v1
        vop3 136 1111 v_subrev_u32_e64 D:v1 S0:v1 S1:v1
        vop3 137 0010 v_dot2c_f32_f16_e64 D:v1 S0:v1 S1:v1
        vop3 138 0010 v_dot2c_i32_i16_e64 D:v1 S0:v1 S1:v1
        vop3 139 0010 v_dot4c_i32_i8_e64 D:v1 S0:v1 S1:v1
        vop3 13a 0010 v_dot8c_i32_i4_e64 D:v1 S0:v1 S1:v1
        vop3 13b 0110 v_fmac_f32_e64 D:v1 S0:v1 S1:v1
        vop3 13d 0110 v_xnor_b32_e64 D:v1 S0:v1 S1:v1
        vop3 140 1111 v_nop -
        vop3 141 1111 v_mov_b32_e64 D:v1 S0:v1
        vop3 143 1111 v_cvt_i32_f64_e64 D:v1 S0:v2
        vop3 144 1111 v_cvt_f64_i32_e64 D:v2 S0:v1
        vop3 145 1111 v_cvt_f32_i32_e64 D:v1 S0:v1
        vop3 146 1111 v_cvt_f32_u32_e64 D:v1 S0:v1
        vop3 147 1111 v_cvt_u32_f32_e64 D:v1 S0:v1
        vop3 148 1111 v_cvt_i32_f32_e64 D:v1 S0:v1
        vop3 14a 1111 v_cvt_f16_f32_e64 D:v1 S0:v1
        vop3 14b 1111 v_cvt_f32_f16_e64 D:v1 S0:v1
        vop3 14c 1111 v_cvt_rpi_i32_f32_e64 D:v1 S0:v1
        vop3 14d 1111 v_cvt_flr_i32_f32_e64 D:v1 S0:v1
        vop3 14e 1111 v_cvt_off_f32_i4_e64 D:v1 S0:v1
        vop3 14f 1111 v_cvt_f32_f64_e64 D:v1 S0:v2
        vop3 150 1111 v_cvt_f64_f32_e64 D:v2 S0:v1
        vop3 151 1111 v_cvt_f32_ubyte0_e64 D:v1 S0:v1
        vop3 152 1111 v_cvt_f32_ubyte1_e64 D:v1 S0:v1
        vop3 153 1111 v_cvt_f32_ubyte2_e64 D:v1 S0:v1
        vop3 154 1111 v_cvt_f32_ubyte3_e64 D:v1 S0:v1
        vop3 155 1111 v_cvt_u32_f64_e64 D:v1 S0:v2
        vop3 156 1111 v_cvt_f64_u32_e64 D:v2 S0:v1
        vop3 157 1111 v_trunc_f64_e64 D:v2 S0:v2
        vop3 158 1111 v_ceil_f64_e64 D:v2 S0:v2
        vop3 159 1111 v_rndne_f64_e64 D:v2 S0:v2
        vop3 15a 1111 v_floor_f64_e64 D:v2 S0:v2
        vop3 15b 1111 v_fract_f32_e64 D:v1 S0:v1
        vop3 15c 1111 v_trunc_f32_e64 D:v1 S0:v1
        vop3 15d 1111 v_ceil_f32_e64 D:v1 S0:v1
        vop3 15e 1111 v_rndne_f32_e64 D:v1 S0:v1
        vop3 15f 1111 v_floor_f32_e64 D:v1 S0:v1
        vop3 160 1111 v_exp_f32_e64 D:v1 S0:v1
        vop3 161 1111 v_log_f32_e64 D:v1 S0:v1
        vop3 162 1111 v_rcp_f32_e64 D:v1 S0:v1
        vop3 163 1111 v_rcp_iflag_f32_e64 D:v1 S0:v1
        vop3 164 1111 v_rsq_f32_e64 D:v1 S0:v1
        vop3 165 1111 v_rcp_f64_e64 D:v2 S0:v2
        vop3 166 1111 v_rsq_f64_e64 D:v2 S0:v2
        vop3 167 1111 v_sqrt_f32_e64 D:v1 S0:v1
        vop3 168 1111 v_sqrt_f64_e64 D:v2 S0:v2
        vop3 169 1111 v_sin_f32_e64 D:v1 S0:v1
        vop3 16a 1111 v_cos_f32_e64 D:v1 S0:v1
        vop3 16b 1111 v_not_b32_e64 D:v1 S0:v1
        vop3 16c 1111 v_bfrev_b32_e64 D:v1 S0:v1
        vop3 16d 1111 v_ffbh_u32_e64 D:v1 S0:v1
        vop3 16e 1111 v_ffbl_b32_e64 D:v1 S0:v1
        vop3 16f 1111 v_ffbh_i32_e64 D:v1 S0:v1
        vop3 170 1111 v_frexp_exp_i32_f64_e64 D:v1 S0:v2
        vop3 171 1111 v_frexp_mant_f64_e64 D:v2 S0:v2
        vop3 172 1111 v_fract_f64_e64 D:v2 S0:v2
        vop3 173 1111 v_frexp_exp_i32_f32_e64 D:v1 S0:v1
        vop3 174 1111 v_frexp_mant_f32_e64 D:v1 S0:v1
        vop3 175 1111 v_clrexcp -
        vop3 177 1111 v_screen_partition_4se_b32_e64 D:v1 S0:v1
        vop3 179 1111 v_cvt_f16_u16_e64 D:v1 S0:v1
        vop3 17a 1111 v_cvt_f16_i16_e64 D:v1 S0:v1
        vop3 17b 1111 v_cvt_u16_f16_e64 D:v1 S0:v1
        vop3 17c 1111 v_cvt_i16_f16_e64 D:v1 S0:v1
        vop3 17d 1111 v_rcp_f16_e64 D:v1 S0:v1
        vop3 17e 1111 v_sqrt_f16_e64 D:v1 S0:v1
        vop3 17f 1111 v_rsq_f16_e64 D:v1 S0:v1
        vop3 180 1111 v_log_f16_e64 D:v1 S0:v1
        vop3 181 1111 v_exp_f16_e64 D:v1 S0:v1
        vop3 182 1111 v_frexp_mant_f16_e64 D:v1 S0:v1
        vop3 183 1111 v_frexp_exp_i16_f16_e64 D:v1 S0:v1
        vop3 184 1111 v_floor_f16_e64 D:v1 S0:v1
        vop3 185 1111 v_ceil_f16_e64 D:v1 S0:v1
        vop3 186 1111 v_trunc_f16_e64 D:v1 S0:v1
        vop3 187 1111 v_rndne_f16_e64 D:v1 S0:v1
        vop3 188 1111 v_fract_f16_e64 D:v1 S0:v1
        vop3 189 1111 v_sin_f16_e64 D:v1 S0:v1
        vop3 18a 1111 v_cos_f16_e64 D:v1 S0:v1
        vop3 18b 1111 v_exp_legacy_f32_e64 D:v1 S0:v1
        vop3 18c 1111 v_log_legacy_f32_e64 D:v1 S0:v1
        vop3 18d 1111 v_cvt_norm_i16_f16_e64 D:v1 S0:v1
        vop3 18e 1111 v_cvt_norm_u16_f16_e64 D:v1 S0:v1
        vop3 18f 1111 v_sat_pk_u8_i16_e64 D:v1 S0:v1
        vop3 1c0 1111 v_mad_legacy_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1c1 1111 v_mad_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1c2 1111 v_mad_i32_i24 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1c3 1111 v_mad_u32_u24 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1c4 1111 v_cubeid_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1c5 1111 v_cubesc_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1c6 1111 v_cubetc_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1c7 1111 v_cubema_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1c8 1111 v_bfe_u32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1c9 1111 v_bfe_i32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1ca 1111 v_bfi_b32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1cb 1111 v_fma_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1cc 1111 v_fma_f64 D:v2 S0:v2 S1:v2 S2:v2
        vop3 1cd 1111 v_lerp_u8 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1ce 1111 v_alignbit_b32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1cf 1111 v_alignbyte_b32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1d0 1111 v_min3_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1d1 1111 v_min3_i32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1d2 1111 v_min3_u32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1d3 1111 v_max3_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1d4 1111 v_max3_i32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1d5 1111 v_max3_u32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1d6 1111 v_med3_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1d7 1111 v_med3_i32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1d8 1111 v_med3_u32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1d9 1111 v_sad_u8 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1da 1111 v_sad_hi_u8 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1db 1111 v_sad_u16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1dc 1111 v_sad_u32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1dd 1111 v_cvt_pk_u8_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1de 1111 v_div_fixup_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1df 1111 v_div_fixup_f64 D:v2 S0:v2 S1:v2 S2:v2
        vop3 1e0 1111 v_div_scale_f32 D:v1 SD:s2 S0:v1 S1:v1 S2:v1
        vop3 1e1 1111 v_div_scale_f64 D:v2 SD:s2 S0:v2 S1:v2 S2:v2
        vop3 1e2 1111 v_div_fmas_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1e3 1111 v_div_fmas_f64 D:v2 S0:v2 S1:v2 S2:v2
        vop3 1e4 1111 v_msad_u8 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1e5 1111 v_qsad_pk_u16_u8 D:v2 S0:v2 S1:v1 S2:v2
        vop3 1e6 1111 v_mqsad_pk_u16_u8 D:v2 S0:v2 S1:v1 S2:v2
        vop3 1e7 1111 v_mqsad_u32_u8 D:v4 S0:v2 S1:v1 S2:v4
        vop3 1e8 1111 v_mad_u64_u32 D:v2 SD:s2 S0:v1 S1:v1 S2:v2
        vop3 1e9 1111 v_mad_i64_i32 D:v2 SD:s2 S0:v1 S1:v1 S2:v2
        vop3 1ea 1111 v_mad_legacy_f16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1eb 1111 v_mad_legacy_u16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1ec 1111 v_mad_legacy_i16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1ed 1111 v_perm_b32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1ee 1111 v_fma_legacy_f16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1ef 1111 v_div_fixup_legacy_f16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1f0 1111 v_cvt_pkaccum_u8_f32 D:v1 S0:v1 S1:v1
        vop3 1f1 1111 v_mad_u32_u16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1f2 1111 v_mad_i32_i16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1f3 1111 v_xad_u32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1f4 1111 v_min3_f16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1f5 1111 v_min3_i16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1f6 1111 v_min3_u16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1f7 1111 v_max3_f16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1f8 1111 v_max3_i16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1f9 1111 v_max3_u16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1fa 1111 v_med3_f16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1fb 1111 v_med3_i16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1fc 1111 v_med3_u16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1fd 1111 v_lshl_add_u32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1fe 1111 v_add_lshl_u32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 1ff 1111 v_add3_u32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 200 1111 v_lshl_or_b32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 201 1111 v_and_or_b32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 202 1111 v_or3_b32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 203 1111 v_mad_f16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 204 1111 v_mad_u16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 205 1111 v_mad_i16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 206 1111 v_fma_f16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 207 1111 v_div_fixup_f16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 280 1111 v_add_f64 D:v2 S0:v2 S1:v2
        vop3 281 1111 v_mul_f64 D:v2 S0:v2 S1:v2
        vop3 282 1111 v_min_f64 D:v2 S0:v2 S1:v2
        vop3 283 1111 v_max_f64 D:v2 S0:v2 S1:v2
        vop3 284 1111 v_ldexp_f64 D:v2 S0:v2 S1:v1
        vop3 285 1111 v_mul_lo_u32 D:v1 S0:v1 S1:v1
        vop3 286 1111 v_mul_hi_u32 D:v1 S0:v1 S1:v1
        vop3 287 1111 v_mul_hi_i32 D:v1 S0:v1 S1:v1
        vop3 288 1111 v_ldexp_f32 D:v1 S0:v1 S1:v1
        vop3 289 1111 v_readlane_b32 D:s1 S0:v1 S1:s1
        vop3 28a 1111 v_writelane_b32 D:v1 S0:s1 S1:s1
        vop3 28b 1111 v_bcnt_u32_b32 D:v1 S0:v1 S1:v1
        vop3 28c 1111 v_mbcnt_lo_u32_b32 D:v1 S0:v1 S1:v1
        vop3 28d 1111 v_mbcnt_hi_u32_b32 D:v1 S0:v1 S1:v1
        vop3 28f 1111 v_lshlrev_b64 D:v2 S0:v1 S1:v2
        vop3 290 1111 v_lshrrev_b64 D:v2 S0:v1 S1:v2
        vop3 291 1111 v_ashrrev_i64 D:v2 S0:v1 S1:v2
        vop3 292 1111 v_trig_preop_f64 D:v2 S0:v2 S1:v1
        vop3 293 1111 v_bfm_b32 D:v1 S0:v1 S1:v1
        vop3 294 1111 v_cvt_pknorm_i16_f32 D:v1 S0:v1 S1:v1
        vop3 295 1111 v_cvt_pknorm_u16_f32 D:v1 S0:v1 S1:v1
        vop3 296 1111 v_cvt_pkrtz_f16_f32 D:v1 S0:v1 S1:v1
        vop3 297 1111 v_cvt_pk_u16_u32 D:v1 S0:v1 S1:v1
        vop3 298 1111 v_cvt_pk_i16_i32 D:v1 S0:v1 S1:v1
        vop3 299 1111 v_cvt_pknorm_i16_f16 D:v1 S0:v1 S1:v1
        vop3 29a 1111 v_cvt_pknorm_u16_f16 D:v1 S0:v1 S1:v1
        vop3 29c 1111 v_add_i32 D:v1 S0:v1 S1:v1
        vop3 29d 1111 v_sub_i32 D:v1 S0:v1 S1:v1
        vop3 29e 1111 v_add_i16 D:v1 S0:v1 S1:v1
        vop3 29f 1111 v_sub_i16 D:v1 S0:v1 S1:v1
        vop3 2a0 1111 v_pack_b32_f16 D:v1 S0:v1 S1:v1
        vop3p 000 1111 v_pk_mad_i16 D:v1 S0:v1 S1:v1 S2:v1
        vop3p 001 1111 v_pk_mul_lo_u16 D:v1 S0:v1 S1:v1
        vop3p 002 1111 v_pk_add_i16 D:v1 S0:v1 S1:v1
        vop3p 003 1111 v_pk_sub_i16 D:v1 S0:v1 S1:v1
        vop3p 004 1111 v_pk_lshlrev_b16 D:v1 S0:v1 S1:v1
        vop3p 005 1111 v_pk_lshrrev_b16 D:v1 S0:v1 S1:v1
        vop3p 006 1111 v_pk_ashrrev_i16 D:v1 S0:v1 S1:v1
        vop3p 007 1111 v_pk_max_i16 D:v1 S0:v1 S1:v1
        vop3p 008 1111 v_pk_min_i16 D:v1 S0:v1 S1:v1
        vop3p 009 1111 v_pk_mad_u16 D:v1 S0:v1 S1:v1 S2:v1
        vop3p 00a 1111 v_pk_add_u16 D:v1 S0:v1 S1:v1
        vop3p 00b 1111 v_pk_sub_u16 D:v1 S0:v1 S1:v1
        vop3p 00c 1111 v_pk_max_u16 D:v1 S0:v1 S1:v1
        vop3p 00d 1111 v_pk_min_u16 D:v1 S0:v1 S1:v1
        vop3p 00e 1111 v_pk_fma_f16 D:v1 S0:v1 S1:v1 S2:v1
        vop3p 00f 1111 v_pk_add_f16 D:v1 S0:v1 S1:v1
        vop3p 010 1111 v_pk_mul_f16 D:v1 S0:v1 S1:v1
        vop3p 011 1111 v_pk_min_f16 D:v1 S0:v1 S1:v1
        vop3p 012 1111 v_pk_max_f16 D:v1 S0:v1 S1:v1
        vop3p 020 0110 v_fma_mix_f32 D:v1 S0:v1 S1:v1 S2:v1 mix
        vop3p 020 1001 v_mad_mix_f32 D:v1 S0:v1 S1:v1 S2:v1 mix
        vop3p 021 0110 v_fma_mixlo_f16 D:v1 S0:v1 S1:v1 S2:v1 mix
        vop3p 021 1001 v_mad_mixlo_f16 D:v1 S0:v1 S1:v1 S2:v1 mix
        vop3p 022 0110 v_fma_mixhi_f16 D:v1 S0:v1 S1:v1 S2:v1 mix
        vop3p 022 1001 v_mad_mixhi_f16 D:v1 S0:v1 S1:v1 S2:v1 mix
        vop3p 023 0110 v_dot2_f32_f16 D:v1 S0:v1 S1:v1 S2:v1
        vop3p 026 0110 v_dot2_i32_i16 D:v1 S0:v1 S1:v1 S2:v1
        vop3p 027 0110 v_dot2_u32_u16 D:v1 S0:v1 S1:v1 S2:v1
        vop3p 028 0110 v_dot4_i32_i8 D:v1 S0:v1 S1:v1 S2:v1
        vop3p 029 0110 v_dot4_u32_u8 D:v1 S0:v1 S1:v1 S2:v1
        vop3p 02a 0110 v_dot8_i32_i4 D:v1 S0:v1 S1:v1 S2:v1
        vop3p 02b 0110 v_dot8_u32_u4 D:v1 S0:v1 S1:v1 S2:v1
        vop3p 058 0010 v_accvgpr_read_b32 D:v1 S0:a1
        vop3p 059 0010 v_accvgpr_write_b32 D:a1 S0:v1
        ds 000 1111 ds_add_u32 A:v1 S0:v1 O:16
        ds 001 1111 ds_sub_u32 A:v1 S0:v1 O:16
        ds 002 1111 ds_rsub_u32 A:v1 S0:v1 O:16
        ds 003 1111 ds_inc_u32 A:v1 S0:v1 O:16
        ds 004 1111 ds_dec_u32 A:v1 S0:v1 O:16
        ds 005 1111 ds_min_i32 A:v1 S0:v1 O:16
        ds 006 1111 ds_max_i32 A:v1 S0:v1 O:16
        ds 007 1111 ds_min_u32 A:v1 S0:v1 O:16
        ds 008 1111 ds_max_u32 A:v1 S0:v1 O:16
        ds 009 1111 ds_and_b32 A:v1 S0:v1 O:16
        ds 00a 1111 ds_or_b32 A:v1 S0:v1 O:16
        ds 00b 1111 ds_xor_b32 A:v1 S0:v1 O:16
        ds 00c 1111 ds_mskor_b32 A:v1 S0:v1 S1:v1 O:16
        ds 00d 1111 ds_write_b32 A:v1 S0:v1 O:16
        ds 00e 1111 ds_write2_b32 A:v1 S0:v1 S1:v1 O:8x2
        ds 00f 1111 ds_write2st64_b32 A:v1 S0:v1 S1:v1 O:8x2
        ds 010 1111 ds_cmpst_b32 A:v1 S0:v1 S1:v1 O:16
        ds 011 1111 ds_cmpst_f32 A:v1 S0:v1 S1:v1 O:16
        ds 012 1111 ds_min_f32 A:v1 S0:v1 O:16
        ds 013 1111 ds_max_f32 A:v1 S0:v1 O:16
        ds 015 1111 ds_add_f32 A:v1 S0:v1 O:16
        ds 01d 1111 ds_write_addtid_b32 S0:v1 O:16
        ds 01e 1111 ds_write_b8 A:v1 S0:v1 O:16
        ds 01f 1111 ds_write_b16 A:v1 S0:v1 O:16
        ds 020 1111 ds_add_rtn_u32 D:v1 A:v1 S0:v1 O:16
        ds 021 1111 ds_sub_rtn_u32 D:v1 A:v1 S0:v1 O:16
        ds 022 1111 ds_rsub_rtn_u32 D:v1 A:v1 S0:v1 O:16
        ds 023 1111 ds_inc_rtn_u32 D:v1 A:v1 S0:v1 O:16
        ds 024 1111 ds_dec_rtn_u32 D:v1 A:v1 S0:v1 O:16
        ds 025 1111 ds_min_rtn_i32 D:v1 A:v1 S0:v1 O:16
        ds 026 1111 ds_max_rtn_i32 D:v1 A:v1 S0:v1 O:16
        ds 027 1111 ds_min_rtn_u32 D:v1 A:v1 S0:v1 O:16
        ds 028 1111 ds_max_rtn_u32 D:v1 A:v1 S0:v1 O:16
        ds 029 1111 ds_and_rtn_b32 D:v1 A:v1 S0:v1 O:16
        ds 02a 1111 ds_or_rtn_b32 D:v1 A:v1 S0:v1 O:16
        ds 02b 1111 ds_xor_rtn_b32 D:v1 A:v1 S0:v1 O:16
        ds 02c 1111 ds_mskor_rtn_b32 D:v1 A:v1 S0:v1 S1:v1 O:16
        ds 02d 1111 ds_wrxchg_rtn_b32 D:v1 A:v1 S0:v1 O:16
        ds 02e 1111 ds_wrxchg2_rtn_b32 D:v2 A:v1 S0:v1 S1:v1 O:8x2
        ds 02f 1111 ds_wrxchg2st64_rtn_b32 D:v2 A:v1 S0:v1 S1:v1 O:8x2
        ds 030 1111 ds_cmpst_rtn_b32 D:v1 A:v1 S0:v1 S1:v1 O:16
        ds 031 1111 ds_cmpst_rtn_f32 D:v1 A:v1 S0:v1 S1:v1 O:16
        ds 032 1111 ds_min_rtn_f32 D:v1 A:v1 S0:v1 O:16
        ds 033 1111 ds_max_rtn_f32 D:v1 A:v1 S0:v1 O:16
        ds 034 1111 ds_wrap_rtn_b32 D:v1 A:v1 S0:v1 S1:v1 O:16
        ds 035 1111 ds_add_rtn_f32 D:v1 A:v1 S0:v1 O:16
        ds 036 1111 ds_read_b32 D:v1 A:v1 O:16
        ds 037 1111 ds_read2_b32 D:v2 A:v1 O:8x2
        ds 038 1111 ds_read2st64_b32 D:v2 A:v1 O:8x2
        ds 039 1111 ds_read_i8 D:v1 A:v1 O:16
        ds 03a 1111 ds_read_u8 D:v1 A:v1 O:16
        ds 03b 1111 ds_read_i16 D:v1 A:v1 O:16
        ds 03c 1111 ds_read_u16 D:v1 A:v1 O:16
        ds 03d 1111 ds_swizzle_b32 D:v1 A:v1 O:swz
        ds 03e 1111 ds_permute_b32 D:v1 A:v1 S0:v1 O:16
        ds 03f 1111 ds_bpermute_b32 D:v1 A:v1 S0:v1 O:16
        ds 040 1111 ds_add_u64 A:v1 S0:v2 O:16
        ds 041 1111 ds_sub_u64 A:v1 S0:v2 O:16
        ds 042 1111 ds_rsub_u64 A:v1 S0:v2 O:16
        ds 043 1111 ds_inc_u64 A:v1 S0:v2 O:16
        ds 044 1111 ds_dec_u64 A:v1 S0:v2 O:16
        ds 045 1111 ds_min_i64 A:v1 S0:v2 O:16
        ds 046 1111 ds_max_i64 A:v1 S0:v2 O:16
        ds 047 1111 ds_min_u64 A:v1 S0:v2 O:16
        ds 048 1111 ds_max_u64 A:v1 S0:v2 O:16
        ds 049 1111 ds_and_b64 A:v1 S0:v2 O:16
        ds 04a 1111 ds_or_b64 A:v1 S0:v2 O:16
        ds 04b 1111 ds_xor_b64 A:v1 S0:v2 O:16
        ds 04c 1111 ds_mskor_b64 A:v1 S0:v2 S1:v2 O:16
        ds 04d 1111 ds_write_b64 A:v1 S0:v2 O:16
        ds 04e 1111 ds_write2_b64 A:v1 S0:v2 S1:v2 O:8x2
        ds 04f 1111 ds_write2st64_b64 A:v1 S0:v2 S1:v2 O:8x2
        ds 050 1111 ds_cmpst_b64 A:v1 S0:v2 S1:v2 O:16
        ds 051 1111 ds_cmpst_f64 A:v1 S0:v2 S1:v2 O:16
        ds 052 1111 ds_min_f64 A:v1 S0:v2 O:16
        ds 053 1111 ds_max_f64 A:v1 S0:v2 O:16
        ds 054 1111 ds_write_b8_d16_hi A:v1 S0:v1 O:16
        ds 055 1111 ds_write_b16_d16_hi A:v1 S0:v1 O:16
        ds 056 1111 ds_read_u8_d16 D:v1 A:v1 O:16
        ds 057 1111 ds_read_u8_d16_hi D:v1 A:v1 O:16
        ds 058 1111 ds_read_i8_d16 D:v1 A:v1 O:16
        ds 059 1111 ds_read_i8_d16_hi D:v1 A:v1 O:16
        ds 05a 1111 ds_read_u16_d16 D:v1 A:v1 O:16
        ds 05b 1111 ds_read_u16_d16_hi D:v1 A:v1 O:16
        ds 060 1111 ds_add_rtn_u64 D:v2 A:v1 S0:v2 O:16
        ds 061 1111 ds_sub_rtn_u64 D:v2 A:v1 S0:v2 O:16
        ds 062 1111 ds_rsub_rtn_u64 D:v2 A:v1 S0:v2 O:16
        ds 063 1111 ds_inc_rtn_u64 D:v2 A:v1 S0:v2 O:16
        ds 064 1111 ds_dec_rtn_u64 D:v2 A:v1 S0:v2 O:16
        ds 065 1111 ds_min_rtn_i64 D:v2 A:v1 S0:v2 O:16
        ds 066 1111 ds_max_rtn_i64 D:v2 A:v1 S0:v2 O:16
        ds 067 1111 ds_min_rtn_u64 D:v2 A:v1 S0:v2 O:16
        ds 068 1111 ds_max_rtn_u64 D:v2 A:v1 S0:v2 O:16
        ds 069 1111 ds_and_rtn_b64 D:v2 A:v1 S0:v2 O:16
        ds 06a 1111 ds_or_rtn_b64 D:v2 A:v1 S0:v2 O:16
        ds 06b 1111 ds_xor_rtn_b64 D:v2 A:v1 S0:v2 O:16
        ds 06c 1111 ds_mskor_rtn_b64 D:v2 A:v1 S0:v2 S1:v2 O:16
        ds 06d 1111 ds_wrxchg_rtn_b64 D:v2 A:v1 S0:v2 O:16
        ds 06e 1111 ds_wrxchg2_rtn_b64 D:v4 A:v1 S0:v2 S1:v2 O:8x2
        ds 06f 1111 ds_wrxchg2st64_rtn_b64 D:v4 A:v1 S0:v2 S1:v2 O:8x2
        ds 070 1111 ds_cmpst_rtn_b64 D:v2 A:v1 S0:v2 S1:v2 O:16
        ds 071 1111 ds_cmpst_rtn_f64 D:v2 A:v1 S0:v2 S1:v2 O:16
        ds 072 1111 ds_min_rtn_f64 D:v2 A:v1 S0:v2 O:16
        ds 073 1111 ds_max_rtn_f64 D:v2 A:v1 S0:v2 O:16
        ds 076 1111 ds_read_b64 D:v2 A:v1 O:16
        ds 077 1111 ds_read2_b64 D:v4 A:v1 O:8x2
        ds 078 1111 ds_read2st64_b64 D:v4 A:v1 O:8x2
        ds 07e 1111 ds_condxchg32_rtn_b64 D:v2 A:v1 S0:v2 O:16
        ds 080 1111 ds_add_src2_u32 A:v1 O:16
        ds 081 1111 ds_sub_src2_u32 A:v1 O:16
        ds 082 1111 ds_rsub_src2_u32 A:v1 O:16
        ds 083 1111 ds_inc_src2_u32 A:v1 O:16
        ds 084 1111 ds_dec_src2_u32 A:v1 O:16
        ds 085 1111 ds_min_src2_i32 A:v1 O:16
        ds 086 1111 ds_max_src2_i32 A:v1 O:16
        ds 087 1111 ds_min_src2_u32 A:v1 O:16
        ds 088 1111 ds_max_src2_u32 A:v1 O:16
        ds 089 1111 ds_and_src2_b32 A:v1 O:16
        ds 08a 1111 ds_or_src2_b32 A:v1 O:16
        ds 08b 1111 ds_xor_src2_b32 A:v1 O:16
        ds 08d 1111 ds_write_src2_b32 A:v1 O:16
        ds 092 1111 ds_min_src2_f32 A:v1 O:16
        ds 093 1111 ds_max_src2_f32 A:v1 O:16
        ds 095 1111 ds_add_src2_f32 A:v1 O:16
        ds 0b6 1111 ds_read_addtid_b32 D:v1 O:16
        ds 0bd 1111 ds_consume D:v1 O:16
        ds 0be 1111 ds_append D:v1 O:16
        ds 0c0 1111 ds_add_src2_u64 A:v1 O:16
        ds 0c1 1111 ds_sub_src2_u64 A:v1 O:16
        ds 0c2 1111 ds_rsub_src2_u64 A:v1 O:16
        ds 0c3 1111 ds_inc_src2_u64 A:v1 O:16
        ds 0c4 1111 ds_dec_src2_u64 A:v1 O:16
        ds 0c5 1111 ds_min_src2_i64 A:v1 O:16
        ds 0c6 1111 ds_max_src2_i64 A:v1 O:16
        ds 0c7 1111 ds_min_src2_u64 A:v1 O:16
        ds 0c8 1111 ds_max_src2_u64 A:v1 O:16
        ds 0c9 1111 ds_and_src2_b64 A:v1 O:16
        ds 0ca 1111 ds_or_src2_b64 A:v1 O:16
        ds 0cb 1111 ds_xor_src2_b64 A:v1 O:16
        ds 0cd 1111 ds_write_src2_b64 A:v1 O:16
        ds 0d2 1111 ds_min_src2_f64 A:v1 O:16
        ds 0d3 1111 ds_max_src2_f64 A:v1 O:16
        ds 0de 1111 ds_write_b96 A:v1 S0:v3 O:16
        ds 0df 1111 ds_write_b128 A:v1 S0:v4 O:16
        ds 0fe 1111 ds_read_b96 D:v3 A:v1 O:16
        ds 0ff 1111 ds_read_b128 D:v4 A:v1 O:16
        flat 010 1111 flat_load_ubyte D:v1 A:v2 O:16
        flat 011 1111 flat_load_sbyte D:v1 A:v2 O:16
        flat 012 1111 flat_load_ushort D:v1 A:v2 O:16
        flat 013 1111 flat_load_sshort D:v1 A:v2 O:16
        flat 014 1111 flat_load_dword D:v1 A:v2 O:16
        flat 015 1111 flat_load_dwordx2 D:v2 A:v2 O:16
        flat 016 1111 flat_load_dwordx3 D:v3 A:v2 O:16
        flat 017 1111 flat_load_dwordx4 D:v4 A:v2 O:16
        flat 018 1111 flat_store_byte A:v2 S0:v1 O:16
        flat 019 1111 flat_store_byte_d16_hi A:v2 S0:v1 O:16
        flat 01a 1111 flat_store_short A:v2 S0:v1 O:16
        flat 01b 1111 flat_store_short_d16_hi A:v2 S0:v1 O:16
        flat 01c 1111 flat_store_dword A:v2 S0:v1 O:16
        flat 01d 1111 flat_store_dwordx2 A:v2 S0:v2 O:16
        flat 01e 1111 flat_store_dwordx3 A:v2 S0:v3 O:16
        flat 01f 1111 flat_store_dwordx4 A:v2 S0:v4 O:16
        flat 020 1111 flat_load_ubyte_d16 D:v1 A:v2 O:16
        flat 021 1111 flat_load_ubyte_d16_hi D:v1 A:v2 O:16
        flat 022 1111 flat_load_sbyte_d16 D:v1 A:v2 O:16
        flat 023 1111 flat_load_sbyte_d16_hi D:v1 A:v2 O:16
        flat 024 1111 flat_load_short_d16 D:v1 A:v2 O:16
        flat 025 1111 flat_load_short_d16_hi D:v1 A:v2 O:16
        flat 040 1111 flat_atomic_swap A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 041 1111 flat_atomic_cmpswap A:v2 S0:v2 O:16 | D:v1 A:v2 S0:v2 O:16
        flat 042 1111 flat_atomic_add A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 043 1111 flat_atomic_sub A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 044 1111 flat_atomic_smin A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 045 1111 flat_atomic_umin A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 046 1111 flat_atomic_smax A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 047 1111 flat_atomic_umax A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 048 1111 flat_atomic_and A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 049 1111 flat_atomic_or A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 04a 1111 flat_atomic_xor A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 04b 1111 flat_atomic_inc A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 04c 1111 flat_atomic_dec A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 060 1111 flat_atomic_swap_x2 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 061 1111 flat_atomic_cmpswap_x2 A:v2 S0:v4 O:16 | D:v2 A:v2 S0:v4 O:16
        flat 062 1111 flat_atomic_add_x2 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 063 1111 flat_atomic_sub_x2 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 064 1111 flat_atomic_smin_x2 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 065 1111 flat_atomic_umin_x2 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 066 1111 flat_atomic_smax_x2 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 067 1111 flat_atomic_umax_x2 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 068 1111 flat_atomic_and_x2 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 069 1111 flat_atomic_or_x2 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 06a 1111 flat_atomic_xor_x2 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 06b 1111 flat_atomic_inc_x2 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 06c 1111 flat_atomic_dec_x2 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        global 010 1111 global_load_ubyte D:v1 A:v2 =off O:16
        global 011 1111 global_load_sbyte D:v1 A:v2 =off O:16
        global 012 1111 global_load_ushort D:v1 A:v2 =off O:16
        global 013 1111 global_load_sshort D:v1 A:v2 =off O:16
        global 014 1111 global_load_dword D:v1 A:v2 =off O:16
        global 015 1111 global_load_dwordx2 D:v2 A:v2 =off O:16
        global 016 1111 global_load_dwordx3 D:v3 A:v2 =off O:16
        global 017 1111 global_load_dwordx4 D:v4 A:v2 =off O:16
        global 018 1111 global_store_byte A:v2 S0:v1 =off O:16
        global 019 1111 global_store_byte_d16_hi A:v2 S0:v1 =off O:16
        global 01a 1111 global_store_short A:v2 S0:v1 =off O:16
        global 01b 1111 global_store_short_d16_hi A:v2 S0:v1 =off O:16
        global 01c 1111 global_store_dword A:v2 S0:v1 =off O:16
        global 01d 1111 global_store_dwordx2 A:v2 S0:v2 =off O:16
        global 01e 1111 global_store_dwordx3 A:v2 S0:v3 =off O:16
        global 01f 1111 global_store_dwordx4 A:v2 S0:v4 =off O:16
        global 020 1111 global_load_ubyte_d16 D:v1 A:v2 =off O:16
        global 021 1111 global_load_ubyte_d16_hi D:v1 A:v2 =off O:16
        global 022 1111 global_load_sbyte_d16 D:v1 A:v2 =off O:16
        global 023 1111 global_load_sbyte_d16_hi D:v1 A:v2 =off O:16
        global 024 1111 global_load_short_d16 D:v1 A:v2 =off O:16
        global 025 1111 global_load_short_d16_hi D:v1 A:v2 =off O:16
        global 040 1111 global_atomic_swap A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 041 1111 global_atomic_cmpswap A:v2 S0:v2 =off O:16 | D:v1 A:v2 S0:v2 =off O:16
        global 042 1111 global_atomic_add A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 043 1111 global_atomic_sub A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 044 1111 global_atomic_smin A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 045 1111 global_atomic_umin A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 046 1111 global_atomic_smax A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 047 1111 global_atomic_umax A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 048 1111 global_atomic_and A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 049 1111 global_atomic_or A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 04a 1111 global_atomic_xor A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 04b 1111 global_atomic_inc A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 04c 1111 global_atomic_dec A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 04d 0010 global_atomic_add_f32 A:v2 S0:v1 =off O:16
        global 04e 0010 global_atomic_pk_add_f16 A:v2 S0:v1 =off O:16
        global 060 1111 global_atomic_swap_x2 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 061 1111 global_atomic_cmpswap_x2 A:v2 S0:v4 =off O:16 | D:v2 A:v2 S0:v4 =off O:16
        global 062 1111 global_atomic_add_x2 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 063 1111 global_atomic_sub_x2 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 064 1111 global_atomic_smin_x2 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 065 1111 global_atomic_umin_x2 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 066 1111 global_atomic_smax_x2 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 067 1111 global_atomic_umax_x2 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 068 1111 global_atomic_and_x2 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 069 1111 global_atomic_or_x2 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 06a 1111 global_atomic_xor_x2 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 06b 1111 global_atomic_inc_x2 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 06c 1111 global_atomic_dec_x2 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        scratch 010 1111 scratch_load_ubyte D:v1 A:v1 =off O:16
        scratch 011 1111 scratch_load_sbyte D:v1 A:v1 =off O:16
        scratch 012 1111 scratch_load_ushort D:v1 A:v1 =off O:16
        scratch 013 1111 scratch_load_sshort D:v1 A:v1 =off O:16
        scratch 014 1111 scratch_load_dword D:v1 A:v1 =off O:16
        scratch 015 1111 scratch_load_dwordx2 D:v2 A:v1 =off O:16
        scratch 016 1111 scratch_load_dwordx3 D:v3 A:v1 =off O:16
        scratch 017 1111 scratch_load_dwordx4 D:v4 A:v1 =off O:16
        scratch 018 1111 scratch_store_byte A:v1 S0:v1 =off O:16
        scratch 019 1111 scratch_store_byte_d16_hi A:v1 S0:v1 =off O:16
        scratch 01a 1111 scratch_store_short A:v1 S0:v1 =off O:16
        scratch 01b 1111 scratch_store_short_d16_hi A:v1 S0:v1 =off O:16
        scratch 01c 1111 scratch_store_dword A:v1 S0:v1 =off O:16
        scratch 01d 1111 scratch_store_dwordx2 A:v1 S0:v2 =off O:16
        scratch 01e 1111 scratch_store_dwordx3 A:v1 S0:v3 =off O:16
        scratch 01f 1111 scratch_store_dwordx4 A:v1 S0:v4 =off O:16
        scratch 020 1111 scratch_load_ubyte_d16 D:v1 A:v1 =off O:16
        scratch 021 1111 scratch_load_ubyte_d16_hi D:v1 A:v1 =off O:16
        scratch 022 1111 scratch_load_sbyte_d16 D:v1 A:v1 =off O:16
        scratch 023 1111 scratch_load_sbyte_d16_hi D:v1 A:v1 =off O:16
        scratch 024 1111 scratch_load_short_d16 D:v1 A:v1 =off O:16
        scratch 025 1111 scratch_load_short_d16_hi D:v1 A:v1 =off O:16
        """;
}
