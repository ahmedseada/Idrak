// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Tests.Gcn;

// Made by tests/Idrak.Tests/data/gcn/interpreter/make_opcode_table.py from LLVM 18.1.8's disassembler; do not edit, rerun the
// script. One opcode a line: the encoding, the opcode (hex), the CPUs that have it (gfx1010, gfx1030, 1 or 0 each), the mnemonic
// as LLVM prints it, and the operand template in print order (a field, its register class and width; '=' a fixed
// operand; K an immediate and its kind; O the offset's kind); after '|', FLAT's template when glc is set.
// The gfx10 family in wave64, in its own word layouts (plan 35, G8a).
internal static partial class GcnOpcodeTable
{
    internal const string Gfx10Cpus = "gfx1010 gfx1030";

    internal const string Gfx10Text = """
        sop1 003 11 s_mov_b32 D:s1 S0:s1
        sop1 004 11 s_mov_b64 D:s2 S0:s2
        sop1 005 11 s_cmov_b32 D:s1 S0:s1
        sop1 006 11 s_cmov_b64 D:s2 S0:s2
        sop1 007 11 s_not_b32 D:s1 S0:s1
        sop1 008 11 s_not_b64 D:s2 S0:s2
        sop1 009 11 s_wqm_b32 D:s1 S0:s1
        sop1 00a 11 s_wqm_b64 D:s2 S0:s2
        sop1 00b 11 s_brev_b32 D:s1 S0:s1
        sop1 00c 11 s_brev_b64 D:s2 S0:s2
        sop1 00d 11 s_bcnt0_i32_b32 D:s1 S0:s1
        sop1 00e 11 s_bcnt0_i32_b64 D:s1 S0:s2
        sop1 00f 11 s_bcnt1_i32_b32 D:s1 S0:s1
        sop1 010 11 s_bcnt1_i32_b64 D:s1 S0:s2
        sop1 011 11 s_ff0_i32_b32 D:s1 S0:s1
        sop1 012 11 s_ff0_i32_b64 D:s1 S0:s2
        sop1 013 11 s_ff1_i32_b32 D:s1 S0:s1
        sop1 014 11 s_ff1_i32_b64 D:s1 S0:s2
        sop1 015 11 s_flbit_i32_b32 D:s1 S0:s1
        sop1 016 11 s_flbit_i32_b64 D:s1 S0:s2
        sop1 017 11 s_flbit_i32 D:s1 S0:s1
        sop1 018 11 s_flbit_i32_i64 D:s1 S0:s2
        sop1 019 11 s_sext_i32_i8 D:s1 S0:s1
        sop1 01a 11 s_sext_i32_i16 D:s1 S0:s1
        sop1 01b 11 s_bitset0_b32 D:s1 S0:s1
        sop1 01c 11 s_bitset0_b64 D:s2 S0:s1
        sop1 01d 11 s_bitset1_b32 D:s1 S0:s1
        sop1 01e 11 s_bitset1_b64 D:s2 S0:s1
        sop1 01f 11 s_getpc_b64 D:s2
        sop1 020 11 s_setpc_b64 S0:s2
        sop1 021 11 s_swappc_b64 D:s2 S0:s2
        sop1 022 11 s_rfe_b64 S0:s2
        sop1 024 11 s_and_saveexec_b64 D:s2 S0:s2
        sop1 025 11 s_or_saveexec_b64 D:s2 S0:s2
        sop1 026 11 s_xor_saveexec_b64 D:s2 S0:s2
        sop1 027 11 s_andn2_saveexec_b64 D:s2 S0:s2
        sop1 028 11 s_orn2_saveexec_b64 D:s2 S0:s2
        sop1 029 11 s_nand_saveexec_b64 D:s2 S0:s2
        sop1 02a 11 s_nor_saveexec_b64 D:s2 S0:s2
        sop1 02b 11 s_xnor_saveexec_b64 D:s2 S0:s2
        sop1 02c 11 s_quadmask_b32 D:s1 S0:s1
        sop1 02d 11 s_quadmask_b64 D:s2 S0:s2
        sop1 02e 11 s_movrels_b32 D:s1 S0:s1
        sop1 02f 11 s_movrels_b64 D:s2 S0:s2
        sop1 030 11 s_movreld_b32 D:s1 S0:s1
        sop1 031 11 s_movreld_b64 D:s2 S0:s2
        sop1 034 11 s_abs_i32 D:s1 S0:s1
        sop1 037 11 s_andn1_saveexec_b64 D:s2 S0:s2
        sop1 038 11 s_orn1_saveexec_b64 D:s2 S0:s2
        sop1 039 11 s_andn1_wrexec_b64 D:s2 S0:s2
        sop1 03a 11 s_andn2_wrexec_b64 D:s2 S0:s2
        sop1 03b 11 s_bitreplicate_b64_b32 D:s2 S0:s1
        sop1 03c 11 s_and_saveexec_b32 D:s1 S0:s1
        sop1 03d 11 s_or_saveexec_b32 D:s1 S0:s1
        sop1 03e 11 s_xor_saveexec_b32 D:s1 S0:s1
        sop1 03f 11 s_andn2_saveexec_b32 D:s1 S0:s1
        sop1 040 11 s_orn2_saveexec_b32 D:s1 S0:s1
        sop1 041 11 s_nand_saveexec_b32 D:s1 S0:s1
        sop1 042 11 s_nor_saveexec_b32 D:s1 S0:s1
        sop1 043 11 s_xnor_saveexec_b32 D:s1 S0:s1
        sop1 044 11 s_andn1_saveexec_b32 D:s1 S0:s1
        sop1 045 11 s_orn1_saveexec_b32 D:s1 S0:s1
        sop1 046 11 s_andn1_wrexec_b32 D:s1 S0:s1
        sop1 047 11 s_andn2_wrexec_b32 D:s1 S0:s1
        sop1 049 11 s_movrelsd_2_b32 D:s1 S0:s1
        sop2 000 11 s_add_u32 D:s1 S0:s1 S1:s1
        sop2 001 11 s_sub_u32 D:s1 S0:s1 S1:s1
        sop2 002 11 s_add_i32 D:s1 S0:s1 S1:s1
        sop2 003 11 s_sub_i32 D:s1 S0:s1 S1:s1
        sop2 004 11 s_addc_u32 D:s1 S0:s1 S1:s1
        sop2 005 11 s_subb_u32 D:s1 S0:s1 S1:s1
        sop2 006 11 s_min_i32 D:s1 S0:s1 S1:s1
        sop2 007 11 s_min_u32 D:s1 S0:s1 S1:s1
        sop2 008 11 s_max_i32 D:s1 S0:s1 S1:s1
        sop2 009 11 s_max_u32 D:s1 S0:s1 S1:s1
        sop2 00a 11 s_cselect_b32 D:s1 S0:s1 S1:s1
        sop2 00b 11 s_cselect_b64 D:s2 S0:s2 S1:s2
        sop2 00e 11 s_and_b32 D:s1 S0:s1 S1:s1
        sop2 00f 11 s_and_b64 D:s2 S0:s2 S1:s2
        sop2 010 11 s_or_b32 D:s1 S0:s1 S1:s1
        sop2 011 11 s_or_b64 D:s2 S0:s2 S1:s2
        sop2 012 11 s_xor_b32 D:s1 S0:s1 S1:s1
        sop2 013 11 s_xor_b64 D:s2 S0:s2 S1:s2
        sop2 014 11 s_andn2_b32 D:s1 S0:s1 S1:s1
        sop2 015 11 s_andn2_b64 D:s2 S0:s2 S1:s2
        sop2 016 11 s_orn2_b32 D:s1 S0:s1 S1:s1
        sop2 017 11 s_orn2_b64 D:s2 S0:s2 S1:s2
        sop2 018 11 s_nand_b32 D:s1 S0:s1 S1:s1
        sop2 019 11 s_nand_b64 D:s2 S0:s2 S1:s2
        sop2 01a 11 s_nor_b32 D:s1 S0:s1 S1:s1
        sop2 01b 11 s_nor_b64 D:s2 S0:s2 S1:s2
        sop2 01c 11 s_xnor_b32 D:s1 S0:s1 S1:s1
        sop2 01d 11 s_xnor_b64 D:s2 S0:s2 S1:s2
        sop2 01e 11 s_lshl_b32 D:s1 S0:s1 S1:s1
        sop2 01f 11 s_lshl_b64 D:s2 S0:s2 S1:s1
        sop2 020 11 s_lshr_b32 D:s1 S0:s1 S1:s1
        sop2 021 11 s_lshr_b64 D:s2 S0:s2 S1:s1
        sop2 022 11 s_ashr_i32 D:s1 S0:s1 S1:s1
        sop2 023 11 s_ashr_i64 D:s2 S0:s2 S1:s1
        sop2 024 11 s_bfm_b32 D:s1 S0:s1 S1:s1
        sop2 025 11 s_bfm_b64 D:s2 S0:s1 S1:s1
        sop2 026 11 s_mul_i32 D:s1 S0:s1 S1:s1
        sop2 027 11 s_bfe_u32 D:s1 S0:s1 S1:s1
        sop2 028 11 s_bfe_i32 D:s1 S0:s1 S1:s1
        sop2 029 11 s_bfe_u64 D:s2 S0:s2 S1:s1
        sop2 02a 11 s_bfe_i64 D:s2 S0:s2 S1:s1
        sop2 02c 11 s_absdiff_i32 D:s1 S0:s1 S1:s1
        sop2 02e 11 s_lshl1_add_u32 D:s1 S0:s1 S1:s1
        sop2 02f 11 s_lshl2_add_u32 D:s1 S0:s1 S1:s1
        sop2 030 11 s_lshl3_add_u32 D:s1 S0:s1 S1:s1
        sop2 031 11 s_lshl4_add_u32 D:s1 S0:s1 S1:s1
        sop2 032 11 s_pack_ll_b32_b16 D:s1 S0:s1 S1:s1
        sop2 033 11 s_pack_lh_b32_b16 D:s1 S0:s1 S1:s1
        sop2 034 11 s_pack_hh_b32_b16 D:s1 S0:s1 S1:s1
        sop2 035 11 s_mul_hi_u32 D:s1 S0:s1 S1:s1
        sop2 036 11 s_mul_hi_i32 D:s1 S0:s1 S1:s1
        sopk 000 11 s_movk_i32 D:s1 K:hex16
        sopk 001 11 s_version K:hex16
        sopk 002 11 s_cmovk_i32 D:s1 K:hex16
        sopk 003 11 s_cmpk_eq_i32 D:s1 K:hex16
        sopk 004 11 s_cmpk_lg_i32 D:s1 K:hex16
        sopk 005 11 s_cmpk_gt_i32 D:s1 K:hex16
        sopk 006 11 s_cmpk_ge_i32 D:s1 K:hex16
        sopk 007 11 s_cmpk_lt_i32 D:s1 K:hex16
        sopk 008 11 s_cmpk_le_i32 D:s1 K:hex16
        sopk 009 11 s_cmpk_eq_u32 D:s1 K:hex16
        sopk 00a 11 s_cmpk_lg_u32 D:s1 K:hex16
        sopk 00b 11 s_cmpk_gt_u32 D:s1 K:hex16
        sopk 00c 11 s_cmpk_ge_u32 D:s1 K:hex16
        sopk 00d 11 s_cmpk_lt_u32 D:s1 K:hex16
        sopk 00e 11 s_cmpk_le_u32 D:s1 K:hex16
        sopk 00f 11 s_addk_i32 D:s1 K:hex16
        sopk 010 11 s_mulk_i32 D:s1 K:hex16
        sopk 012 11 s_getreg_b32 D:s1 K:hwreg
        sopk 013 11 s_setreg_b32 K:hwreg D:s1
        sopk 015 11 s_setreg_imm32_b32 K:hwreg K:lit
        sopk 016 11 s_call_b64 D:s2 K:dec
        sopk 017 11 s_waitcnt_vscnt D:s1 K:hex16
        sopk 018 11 s_waitcnt_vmcnt D:s1 K:hex16
        sopk 019 11 s_waitcnt_expcnt D:s1 K:hex16
        sopk 01a 11 s_waitcnt_lgkmcnt D:s1 K:hex16
        sopk 01b 11 s_subvector_loop_begin D:s1 K:dec
        sopk 01c 11 s_subvector_loop_end D:s1 K:dec
        sopk 01f 11 s_nop K:hex16
        sopc 000 11 s_cmp_eq_i32 S0:s1 S1:s1
        sopc 001 11 s_cmp_lg_i32 S0:s1 S1:s1
        sopc 002 11 s_cmp_gt_i32 S0:s1 S1:s1
        sopc 003 11 s_cmp_ge_i32 S0:s1 S1:s1
        sopc 004 11 s_cmp_lt_i32 S0:s1 S1:s1
        sopc 005 11 s_cmp_le_i32 S0:s1 S1:s1
        sopc 006 11 s_cmp_eq_u32 S0:s1 S1:s1
        sopc 007 11 s_cmp_lg_u32 S0:s1 S1:s1
        sopc 008 11 s_cmp_gt_u32 S0:s1 S1:s1
        sopc 009 11 s_cmp_ge_u32 S0:s1 S1:s1
        sopc 00a 11 s_cmp_lt_u32 S0:s1 S1:s1
        sopc 00b 11 s_cmp_le_u32 S0:s1 S1:s1
        sopc 00c 11 s_bitcmp0_b32 S0:s1 S1:s1
        sopc 00d 11 s_bitcmp1_b32 S0:s1 S1:s1
        sopc 00e 11 s_bitcmp0_b64 S0:s2 S1:s1
        sopc 00f 11 s_bitcmp1_b64 S0:s2 S1:s1
        sopc 012 11 s_cmp_eq_u64 S0:s2 S1:s2
        sopc 013 11 s_cmp_lg_u64 S0:s2 S1:s2
        sopp 000 11 s_nop K:dec
        sopp 001 11 s_endpgm K:dec
        sopp 002 11 s_branch K:dec
        sopp 003 11 s_wakeup -
        sopp 004 11 s_cbranch_scc0 K:dec
        sopp 005 11 s_cbranch_scc1 K:dec
        sopp 006 11 s_cbranch_vccz K:dec
        sopp 007 11 s_cbranch_vccnz K:dec
        sopp 008 11 s_cbranch_execz K:dec
        sopp 009 11 s_cbranch_execnz K:dec
        sopp 00a 11 s_barrier -
        sopp 00b 11 s_setkill K:dec
        sopp 00c 11 s_waitcnt K:waitcnt
        sopp 00d 11 s_sethalt K:dec
        sopp 00e 11 s_sleep K:dec
        sopp 00f 11 s_setprio K:dec
        sopp 010 11 s_sendmsg K:sendmsg
        sopp 011 11 s_sendmsghalt K:sendmsg
        sopp 012 11 s_trap K:dec
        sopp 013 11 s_icache_inv -
        sopp 014 11 s_incperflevel K:dec
        sopp 015 11 s_decperflevel K:dec
        sopp 016 11 s_ttracedata -
        sopp 017 11 s_cbranch_cdbgsys K:dec
        sopp 018 11 s_cbranch_cdbguser K:dec
        sopp 019 11 s_cbranch_cdbgsys_or_user K:dec
        sopp 01a 11 s_cbranch_cdbgsys_and_user K:dec
        sopp 01b 11 s_endpgm_saved -
        sopp 01e 11 s_endpgm_ordered_ps_done -
        sopp 01f 11 s_code_end -
        sopp 020 11 s_inst_prefetch K:hex16
        sopp 021 11 s_clause K:hex16
        sopp 022 11 s_wait_idle -
        sopp 023 11 s_waitcnt_depctr K:hex16
        sopp 024 11 s_round_mode K:hex16
        sopp 025 11 s_denorm_mode K:dec
        sopp 028 11 s_ttracedata_imm K:hex16
        smem 000 11 s_load_dword D:s1 B:s2 O:hex
        smem 001 11 s_load_dwordx2 D:s2 B:s2 O:hex
        smem 002 11 s_load_dwordx4 D:s4 B:s2 O:hex
        smem 003 11 s_load_dwordx8 D:s8 B:s2 O:hex
        smem 004 11 s_load_dwordx16 D:s16 B:s2 O:hex
        smem 005 10 s_scratch_load_dword D:s1 B:s2 O:hex
        smem 006 10 s_scratch_load_dwordx2 D:s2 B:s2 O:hex
        smem 007 10 s_scratch_load_dwordx4 D:s4 B:s2 O:hex
        smem 008 11 s_buffer_load_dword D:s1 B:s4 O:hex
        smem 009 11 s_buffer_load_dwordx2 D:s2 B:s4 O:hex
        smem 00a 11 s_buffer_load_dwordx4 D:s4 B:s4 O:hex
        smem 00b 11 s_buffer_load_dwordx8 D:s8 B:s4 O:hex
        smem 00c 11 s_buffer_load_dwordx16 D:s16 B:s4 O:hex
        smem 010 10 s_store_dword D:s1 B:s2 O:hex
        smem 011 10 s_store_dwordx2 D:s2 B:s2 O:hex
        smem 012 10 s_store_dwordx4 D:s4 B:s2 O:hex
        smem 015 10 s_scratch_store_dword D:s1 B:s2 O:hex
        smem 016 10 s_scratch_store_dwordx2 D:s2 B:s2 O:hex
        smem 017 10 s_scratch_store_dwordx4 D:s4 B:s2 O:hex
        smem 018 10 s_buffer_store_dword D:s1 B:s4 O:hex
        smem 019 10 s_buffer_store_dwordx2 D:s2 B:s4 O:hex
        smem 01a 10 s_buffer_store_dwordx4 D:s4 B:s4 O:hex
        smem 01f 11 s_gl1_inv -
        smem 020 11 s_dcache_inv -
        smem 021 10 s_dcache_wb -
        smem 024 11 s_memtime D:s2
        smem 025 11 s_memrealtime D:s2
        smem 026 11 s_atc_probe K:dec B:s2 O:hex
        smem 027 11 s_atc_probe_buffer K:dec B:s4 O:hex
        smem 028 10 s_dcache_discard B:s2 O:hex
        smem 029 10 s_dcache_discard_x2 B:s2 O:hex
        smem 02a 10 s_get_waveid_in_workgroup D:s1
        smem 040 10 s_buffer_atomic_swap D:s1 B:s4 O:hex
        smem 041 10 s_buffer_atomic_cmpswap D:s2 B:s4 O:hex
        smem 042 10 s_buffer_atomic_add D:s1 B:s4 O:hex
        smem 043 10 s_buffer_atomic_sub D:s1 B:s4 O:hex
        smem 044 10 s_buffer_atomic_smin D:s1 B:s4 O:hex
        smem 045 10 s_buffer_atomic_umin D:s1 B:s4 O:hex
        smem 046 10 s_buffer_atomic_smax D:s1 B:s4 O:hex
        smem 047 10 s_buffer_atomic_umax D:s1 B:s4 O:hex
        smem 048 10 s_buffer_atomic_and D:s1 B:s4 O:hex
        smem 049 10 s_buffer_atomic_or D:s1 B:s4 O:hex
        smem 04a 10 s_buffer_atomic_xor D:s1 B:s4 O:hex
        smem 04b 10 s_buffer_atomic_inc D:s1 B:s4 O:hex
        smem 04c 10 s_buffer_atomic_dec D:s1 B:s4 O:hex
        smem 060 10 s_buffer_atomic_swap_x2 D:s2 B:s4 O:hex
        smem 061 10 s_buffer_atomic_cmpswap_x2 D:s4 B:s4 O:hex
        smem 062 10 s_buffer_atomic_add_x2 D:s2 B:s4 O:hex
        smem 063 10 s_buffer_atomic_sub_x2 D:s2 B:s4 O:hex
        smem 064 10 s_buffer_atomic_smin_x2 D:s2 B:s4 O:hex
        smem 065 10 s_buffer_atomic_umin_x2 D:s2 B:s4 O:hex
        smem 066 10 s_buffer_atomic_smax_x2 D:s2 B:s4 O:hex
        smem 067 10 s_buffer_atomic_umax_x2 D:s2 B:s4 O:hex
        smem 068 10 s_buffer_atomic_and_x2 D:s2 B:s4 O:hex
        smem 069 10 s_buffer_atomic_or_x2 D:s2 B:s4 O:hex
        smem 06a 10 s_buffer_atomic_xor_x2 D:s2 B:s4 O:hex
        smem 06b 10 s_buffer_atomic_inc_x2 D:s2 B:s4 O:hex
        smem 06c 10 s_buffer_atomic_dec_x2 D:s2 B:s4 O:hex
        smem 080 10 s_atomic_swap D:s1 B:s2 O:hex
        smem 081 10 s_atomic_cmpswap D:s2 B:s2 O:hex
        smem 082 10 s_atomic_add D:s1 B:s2 O:hex
        smem 083 10 s_atomic_sub D:s1 B:s2 O:hex
        smem 084 10 s_atomic_smin D:s1 B:s2 O:hex
        smem 085 10 s_atomic_umin D:s1 B:s2 O:hex
        smem 086 10 s_atomic_smax D:s1 B:s2 O:hex
        smem 087 10 s_atomic_umax D:s1 B:s2 O:hex
        smem 088 10 s_atomic_and D:s1 B:s2 O:hex
        smem 089 10 s_atomic_or D:s1 B:s2 O:hex
        smem 08a 10 s_atomic_xor D:s1 B:s2 O:hex
        smem 08b 10 s_atomic_inc D:s1 B:s2 O:hex
        smem 08c 10 s_atomic_dec D:s1 B:s2 O:hex
        smem 0a0 10 s_atomic_swap_x2 D:s2 B:s2 O:hex
        smem 0a1 10 s_atomic_cmpswap_x2 D:s4 B:s2 O:hex
        smem 0a2 10 s_atomic_add_x2 D:s2 B:s2 O:hex
        smem 0a3 10 s_atomic_sub_x2 D:s2 B:s2 O:hex
        smem 0a4 10 s_atomic_smin_x2 D:s2 B:s2 O:hex
        smem 0a5 10 s_atomic_umin_x2 D:s2 B:s2 O:hex
        smem 0a6 10 s_atomic_smax_x2 D:s2 B:s2 O:hex
        smem 0a7 10 s_atomic_umax_x2 D:s2 B:s2 O:hex
        smem 0a8 10 s_atomic_and_x2 D:s2 B:s2 O:hex
        smem 0a9 10 s_atomic_or_x2 D:s2 B:s2 O:hex
        smem 0aa 10 s_atomic_xor_x2 D:s2 B:s2 O:hex
        smem 0ab 10 s_atomic_inc_x2 D:s2 B:s2 O:hex
        smem 0ac 10 s_atomic_dec_x2 D:s2 B:s2 O:hex
        vop1 000 11 v_nop -
        vop1 001 11 v_mov_b32_e32 D:v1 S0:v1
        vop1 002 11 v_readfirstlane_b32 D:s1 S0:v1
        vop1 003 11 v_cvt_i32_f64_e32 D:v1 S0:v2
        vop1 004 11 v_cvt_f64_i32_e32 D:v2 S0:v1
        vop1 005 11 v_cvt_f32_i32_e32 D:v1 S0:v1
        vop1 006 11 v_cvt_f32_u32_e32 D:v1 S0:v1
        vop1 007 11 v_cvt_u32_f32_e32 D:v1 S0:v1
        vop1 008 11 v_cvt_i32_f32_e32 D:v1 S0:v1
        vop1 00a 11 v_cvt_f16_f32_e32 D:v1 S0:v1
        vop1 00b 11 v_cvt_f32_f16_e32 D:v1 S0:v1
        vop1 00c 11 v_cvt_rpi_i32_f32_e32 D:v1 S0:v1
        vop1 00d 11 v_cvt_flr_i32_f32_e32 D:v1 S0:v1
        vop1 00e 11 v_cvt_off_f32_i4_e32 D:v1 S0:v1
        vop1 00f 11 v_cvt_f32_f64_e32 D:v1 S0:v2
        vop1 010 11 v_cvt_f64_f32_e32 D:v2 S0:v1
        vop1 011 11 v_cvt_f32_ubyte0_e32 D:v1 S0:v1
        vop1 012 11 v_cvt_f32_ubyte1_e32 D:v1 S0:v1
        vop1 013 11 v_cvt_f32_ubyte2_e32 D:v1 S0:v1
        vop1 014 11 v_cvt_f32_ubyte3_e32 D:v1 S0:v1
        vop1 015 11 v_cvt_u32_f64_e32 D:v1 S0:v2
        vop1 016 11 v_cvt_f64_u32_e32 D:v2 S0:v1
        vop1 017 11 v_trunc_f64_e32 D:v2 S0:v2
        vop1 018 11 v_ceil_f64_e32 D:v2 S0:v2
        vop1 019 11 v_rndne_f64_e32 D:v2 S0:v2
        vop1 01a 11 v_floor_f64_e32 D:v2 S0:v2
        vop1 01b 11 v_pipeflush -
        vop1 020 11 v_fract_f32_e32 D:v1 S0:v1
        vop1 021 11 v_trunc_f32_e32 D:v1 S0:v1
        vop1 022 11 v_ceil_f32_e32 D:v1 S0:v1
        vop1 023 11 v_rndne_f32_e32 D:v1 S0:v1
        vop1 024 11 v_floor_f32_e32 D:v1 S0:v1
        vop1 025 11 v_exp_f32_e32 D:v1 S0:v1
        vop1 027 11 v_log_f32_e32 D:v1 S0:v1
        vop1 02a 11 v_rcp_f32_e32 D:v1 S0:v1
        vop1 02b 11 v_rcp_iflag_f32_e32 D:v1 S0:v1
        vop1 02e 11 v_rsq_f32_e32 D:v1 S0:v1
        vop1 02f 11 v_rcp_f64_e32 D:v2 S0:v2
        vop1 031 11 v_rsq_f64_e32 D:v2 S0:v2
        vop1 033 11 v_sqrt_f32_e32 D:v1 S0:v1
        vop1 034 11 v_sqrt_f64_e32 D:v2 S0:v2
        vop1 035 11 v_sin_f32_e32 D:v1 S0:v1
        vop1 036 11 v_cos_f32_e32 D:v1 S0:v1
        vop1 037 11 v_not_b32_e32 D:v1 S0:v1
        vop1 038 11 v_bfrev_b32_e32 D:v1 S0:v1
        vop1 039 11 v_ffbh_u32_e32 D:v1 S0:v1
        vop1 03a 11 v_ffbl_b32_e32 D:v1 S0:v1
        vop1 03b 11 v_ffbh_i32_e32 D:v1 S0:v1
        vop1 03c 11 v_frexp_exp_i32_f64_e32 D:v1 S0:v2
        vop1 03d 11 v_frexp_mant_f64_e32 D:v2 S0:v2
        vop1 03e 11 v_fract_f64_e32 D:v2 S0:v2
        vop1 03f 11 v_frexp_exp_i32_f32_e32 D:v1 S0:v1
        vop1 040 11 v_frexp_mant_f32_e32 D:v1 S0:v1
        vop1 041 11 v_clrexcp -
        vop1 042 11 v_movreld_b32_e32 D:v1 S0:v1
        vop1 043 11 v_movrels_b32_e32 D:v1 S0:v1
        vop1 044 11 v_movrelsd_b32_e32 D:v1 S0:v1
        vop1 048 11 v_movrelsd_2_b32_e32 D:v1 S0:v1
        vop1 050 11 v_cvt_f16_u16_e32 D:v1 S0:v1
        vop1 051 11 v_cvt_f16_i16_e32 D:v1 S0:v1
        vop1 052 11 v_cvt_u16_f16_e32 D:v1 S0:v1
        vop1 053 11 v_cvt_i16_f16_e32 D:v1 S0:v1
        vop1 054 11 v_rcp_f16_e32 D:v1 S0:v1
        vop1 055 11 v_sqrt_f16_e32 D:v1 S0:v1
        vop1 056 11 v_rsq_f16_e32 D:v1 S0:v1
        vop1 057 11 v_log_f16_e32 D:v1 S0:v1
        vop1 058 11 v_exp_f16_e32 D:v1 S0:v1
        vop1 059 11 v_frexp_mant_f16_e32 D:v1 S0:v1
        vop1 05a 11 v_frexp_exp_i16_f16_e32 D:v1 S0:v1
        vop1 05b 11 v_floor_f16_e32 D:v1 S0:v1
        vop1 05c 11 v_ceil_f16_e32 D:v1 S0:v1
        vop1 05d 11 v_trunc_f16_e32 D:v1 S0:v1
        vop1 05e 11 v_rndne_f16_e32 D:v1 S0:v1
        vop1 05f 11 v_fract_f16_e32 D:v1 S0:v1
        vop1 060 11 v_sin_f16_e32 D:v1 S0:v1
        vop1 061 11 v_cos_f16_e32 D:v1 S0:v1
        vop1 062 11 v_sat_pk_u8_i16_e32 D:v1 S0:v1
        vop1 063 11 v_cvt_norm_i16_f16_e32 D:v1 S0:v1
        vop1 064 11 v_cvt_norm_u16_f16_e32 D:v1 S0:v1
        vop1 065 11 v_swap_b32 D:v1 S0:v1
        vop1 068 11 v_swaprel_b32 D:v1 S0:v1
        vop2 001 11 v_cndmask_b32_e32 D:v1 S0:v1 S1:v1 =vcc
        vop2 002 01 v_dot2c_f32_f16 D:v1 S0:v1 S1:v1
        vop2 003 11 v_add_f32_e32 D:v1 S0:v1 S1:v1
        vop2 004 11 v_sub_f32_e32 D:v1 S0:v1 S1:v1
        vop2 005 11 v_subrev_f32_e32 D:v1 S0:v1 S1:v1
        vop2 006 01 v_fmac_legacy_f32_e32 D:v1 S0:v1 S1:v1
        vop2 006 10 v_mac_legacy_f32_e32 D:v1 S0:v1 S1:v1
        vop2 007 11 v_mul_legacy_f32_e32 D:v1 S0:v1 S1:v1
        vop2 008 11 v_mul_f32_e32 D:v1 S0:v1 S1:v1
        vop2 009 11 v_mul_i32_i24_e32 D:v1 S0:v1 S1:v1
        vop2 00a 11 v_mul_hi_i32_i24_e32 D:v1 S0:v1 S1:v1
        vop2 00b 11 v_mul_u32_u24_e32 D:v1 S0:v1 S1:v1
        vop2 00c 11 v_mul_hi_u32_u24_e32 D:v1 S0:v1 S1:v1
        vop2 00d 01 v_dot4c_i32_i8 D:v1 S0:v1 S1:v1
        vop2 00f 11 v_min_f32_e32 D:v1 S0:v1 S1:v1
        vop2 010 11 v_max_f32_e32 D:v1 S0:v1 S1:v1
        vop2 011 11 v_min_i32_e32 D:v1 S0:v1 S1:v1
        vop2 012 11 v_max_i32_e32 D:v1 S0:v1 S1:v1
        vop2 013 11 v_min_u32_e32 D:v1 S0:v1 S1:v1
        vop2 014 11 v_max_u32_e32 D:v1 S0:v1 S1:v1
        vop2 016 11 v_lshrrev_b32_e32 D:v1 S0:v1 S1:v1
        vop2 018 11 v_ashrrev_i32_e32 D:v1 S0:v1 S1:v1
        vop2 01a 11 v_lshlrev_b32_e32 D:v1 S0:v1 S1:v1
        vop2 01b 11 v_and_b32_e32 D:v1 S0:v1 S1:v1
        vop2 01c 11 v_or_b32_e32 D:v1 S0:v1 S1:v1
        vop2 01d 11 v_xor_b32_e32 D:v1 S0:v1 S1:v1
        vop2 01e 11 v_xnor_b32_e32 D:v1 S0:v1 S1:v1
        vop2 01f 10 v_mac_f32_e32 D:v1 S0:v1 S1:v1
        vop2 020 10 v_madmk_f32 D:v1 S0:v1 K:lit S1:v1
        vop2 021 10 v_madak_f32 D:v1 S0:v1 S1:v1 K:lit
        vop2 025 11 v_add_nc_u32_e32 D:v1 S0:v1 S1:v1
        vop2 026 11 v_sub_nc_u32_e32 D:v1 S0:v1 S1:v1
        vop2 027 11 v_subrev_nc_u32_e32 D:v1 S0:v1 S1:v1
        vop2 028 11 v_add_co_ci_u32_e32 D:v1 =vcc S0:v1 S1:v1 =vcc
        vop2 029 11 v_sub_co_ci_u32_e32 D:v1 =vcc S0:v1 S1:v1 =vcc
        vop2 02a 11 v_subrev_co_ci_u32_e32 D:v1 =vcc S0:v1 S1:v1 =vcc
        vop2 02b 11 v_fmac_f32_e32 D:v1 S0:v1 S1:v1
        vop2 02c 11 v_fmamk_f32 D:v1 S0:v1 K:lit S1:v1
        vop2 02d 11 v_fmaak_f32 D:v1 S0:v1 S1:v1 K:lit
        vop2 02f 11 v_cvt_pkrtz_f16_f32_e32 D:v1 S0:v1 S1:v1
        vop2 032 11 v_add_f16_e32 D:v1 S0:v1 S1:v1
        vop2 033 11 v_sub_f16_e32 D:v1 S0:v1 S1:v1
        vop2 034 11 v_subrev_f16_e32 D:v1 S0:v1 S1:v1
        vop2 035 11 v_mul_f16_e32 D:v1 S0:v1 S1:v1
        vop2 036 11 v_fmac_f16_e32 D:v1 S0:v1 S1:v1
        vop2 037 11 v_fmamk_f16 D:v1 S0:v1 K:lit S1:v1
        vop2 038 11 v_fmaak_f16 D:v1 S0:v1 S1:v1 K:lit
        vop2 039 11 v_max_f16_e32 D:v1 S0:v1 S1:v1
        vop2 03a 11 v_min_f16_e32 D:v1 S0:v1 S1:v1
        vop2 03b 11 v_ldexp_f16_e32 D:v1 S0:v1 S1:v1
        vop2 03c 11 v_pk_fmac_f16 D:v1 S0:v1 S1:v1
        vop2 03e 11 v_cmp_nlg_f32_e32 =vcc S0:v1 S1:v1
        vopc 000 11 v_cmp_f_f32_e32 =vcc S0:v1 S1:v1
        vopc 001 11 v_cmp_lt_f32_e32 =vcc S0:v1 S1:v1
        vopc 002 11 v_cmp_eq_f32_e32 =vcc S0:v1 S1:v1
        vopc 003 11 v_cmp_le_f32_e32 =vcc S0:v1 S1:v1
        vopc 004 11 v_cmp_gt_f32_e32 =vcc S0:v1 S1:v1
        vopc 005 11 v_cmp_lg_f32_e32 =vcc S0:v1 S1:v1
        vopc 006 11 v_cmp_ge_f32_e32 =vcc S0:v1 S1:v1
        vopc 007 11 v_cmp_o_f32_e32 =vcc S0:v1 S1:v1
        vopc 008 11 v_cmp_u_f32_e32 =vcc S0:v1 S1:v1
        vopc 009 11 v_cmp_nge_f32_e32 =vcc S0:v1 S1:v1
        vopc 00a 11 v_cmp_nlg_f32_e32 =vcc S0:v1 S1:v1
        vopc 00b 11 v_cmp_ngt_f32_e32 =vcc S0:v1 S1:v1
        vopc 00c 11 v_cmp_nle_f32_e32 =vcc S0:v1 S1:v1
        vopc 00d 11 v_cmp_neq_f32_e32 =vcc S0:v1 S1:v1
        vopc 00e 11 v_cmp_nlt_f32_e32 =vcc S0:v1 S1:v1
        vopc 00f 11 v_cmp_tru_f32_e32 =vcc S0:v1 S1:v1
        vopc 010 11 v_cmpx_f_f32_e32 S0:v1 S1:v1
        vopc 011 11 v_cmpx_lt_f32_e32 S0:v1 S1:v1
        vopc 012 11 v_cmpx_eq_f32_e32 S0:v1 S1:v1
        vopc 013 11 v_cmpx_le_f32_e32 S0:v1 S1:v1
        vopc 014 11 v_cmpx_gt_f32_e32 S0:v1 S1:v1
        vopc 015 11 v_cmpx_lg_f32_e32 S0:v1 S1:v1
        vopc 016 11 v_cmpx_ge_f32_e32 S0:v1 S1:v1
        vopc 017 11 v_cmpx_o_f32_e32 S0:v1 S1:v1
        vopc 018 11 v_cmpx_u_f32_e32 S0:v1 S1:v1
        vopc 019 11 v_cmpx_nge_f32_e32 S0:v1 S1:v1
        vopc 01a 11 v_cmpx_nlg_f32_e32 S0:v1 S1:v1
        vopc 01b 11 v_cmpx_ngt_f32_e32 S0:v1 S1:v1
        vopc 01c 11 v_cmpx_nle_f32_e32 S0:v1 S1:v1
        vopc 01d 11 v_cmpx_neq_f32_e32 S0:v1 S1:v1
        vopc 01e 11 v_cmpx_nlt_f32_e32 S0:v1 S1:v1
        vopc 01f 11 v_cmpx_tru_f32_e32 S0:v1 S1:v1
        vopc 020 11 v_cmp_f_f64_e32 =vcc S0:v2 S1:v2
        vopc 021 11 v_cmp_lt_f64_e32 =vcc S0:v2 S1:v2
        vopc 022 11 v_cmp_eq_f64_e32 =vcc S0:v2 S1:v2
        vopc 023 11 v_cmp_le_f64_e32 =vcc S0:v2 S1:v2
        vopc 024 11 v_cmp_gt_f64_e32 =vcc S0:v2 S1:v2
        vopc 025 11 v_cmp_lg_f64_e32 =vcc S0:v2 S1:v2
        vopc 026 11 v_cmp_ge_f64_e32 =vcc S0:v2 S1:v2
        vopc 027 11 v_cmp_o_f64_e32 =vcc S0:v2 S1:v2
        vopc 028 11 v_cmp_u_f64_e32 =vcc S0:v2 S1:v2
        vopc 029 11 v_cmp_nge_f64_e32 =vcc S0:v2 S1:v2
        vopc 02a 11 v_cmp_nlg_f64_e32 =vcc S0:v2 S1:v2
        vopc 02b 11 v_cmp_ngt_f64_e32 =vcc S0:v2 S1:v2
        vopc 02c 11 v_cmp_nle_f64_e32 =vcc S0:v2 S1:v2
        vopc 02d 11 v_cmp_neq_f64_e32 =vcc S0:v2 S1:v2
        vopc 02e 11 v_cmp_nlt_f64_e32 =vcc S0:v2 S1:v2
        vopc 02f 11 v_cmp_tru_f64_e32 =vcc S0:v2 S1:v2
        vopc 030 11 v_cmpx_f_f64_e32 S0:v2 S1:v2
        vopc 031 11 v_cmpx_lt_f64_e32 S0:v2 S1:v2
        vopc 032 11 v_cmpx_eq_f64_e32 S0:v2 S1:v2
        vopc 033 11 v_cmpx_le_f64_e32 S0:v2 S1:v2
        vopc 034 11 v_cmpx_gt_f64_e32 S0:v2 S1:v2
        vopc 035 11 v_cmpx_lg_f64_e32 S0:v2 S1:v2
        vopc 036 11 v_cmpx_ge_f64_e32 S0:v2 S1:v2
        vopc 037 11 v_cmpx_o_f64_e32 S0:v2 S1:v2
        vopc 038 11 v_cmpx_u_f64_e32 S0:v2 S1:v2
        vopc 039 11 v_cmpx_nge_f64_e32 S0:v2 S1:v2
        vopc 03a 11 v_cmpx_nlg_f64_e32 S0:v2 S1:v2
        vopc 03b 11 v_cmpx_ngt_f64_e32 S0:v2 S1:v2
        vopc 03c 11 v_cmpx_nle_f64_e32 S0:v2 S1:v2
        vopc 03d 11 v_cmpx_neq_f64_e32 S0:v2 S1:v2
        vopc 03e 11 v_cmpx_nlt_f64_e32 S0:v2 S1:v2
        vopc 03f 11 v_cmpx_tru_f64_e32 S0:v2 S1:v2
        vopc 080 11 v_cmp_f_i32_e32 =vcc S0:v1 S1:v1
        vopc 081 11 v_cmp_lt_i32_e32 =vcc S0:v1 S1:v1
        vopc 082 11 v_cmp_eq_i32_e32 =vcc S0:v1 S1:v1
        vopc 083 11 v_cmp_le_i32_e32 =vcc S0:v1 S1:v1
        vopc 084 11 v_cmp_gt_i32_e32 =vcc S0:v1 S1:v1
        vopc 085 11 v_cmp_ne_i32_e32 =vcc S0:v1 S1:v1
        vopc 086 11 v_cmp_ge_i32_e32 =vcc S0:v1 S1:v1
        vopc 087 11 v_cmp_t_i32_e32 =vcc S0:v1 S1:v1
        vopc 088 11 v_cmp_class_f32_e32 =vcc S0:v1 S1:v1
        vopc 089 11 v_cmp_lt_i16_e32 =vcc S0:v1 S1:v1
        vopc 08a 11 v_cmp_eq_i16_e32 =vcc S0:v1 S1:v1
        vopc 08b 11 v_cmp_le_i16_e32 =vcc S0:v1 S1:v1
        vopc 08c 11 v_cmp_gt_i16_e32 =vcc S0:v1 S1:v1
        vopc 08d 11 v_cmp_ne_i16_e32 =vcc S0:v1 S1:v1
        vopc 08e 11 v_cmp_ge_i16_e32 =vcc S0:v1 S1:v1
        vopc 08f 11 v_cmp_class_f16_e32 =vcc S0:v1 S1:v1
        vopc 090 11 v_cmpx_f_i32_e32 S0:v1 S1:v1
        vopc 091 11 v_cmpx_lt_i32_e32 S0:v1 S1:v1
        vopc 092 11 v_cmpx_eq_i32_e32 S0:v1 S1:v1
        vopc 093 11 v_cmpx_le_i32_e32 S0:v1 S1:v1
        vopc 094 11 v_cmpx_gt_i32_e32 S0:v1 S1:v1
        vopc 095 11 v_cmpx_ne_i32_e32 S0:v1 S1:v1
        vopc 096 11 v_cmpx_ge_i32_e32 S0:v1 S1:v1
        vopc 097 11 v_cmpx_t_i32_e32 S0:v1 S1:v1
        vopc 098 11 v_cmpx_class_f32_e32 S0:v1 S1:v1
        vopc 099 11 v_cmpx_lt_i16_e32 S0:v1 S1:v1
        vopc 09a 11 v_cmpx_eq_i16_e32 S0:v1 S1:v1
        vopc 09b 11 v_cmpx_le_i16_e32 S0:v1 S1:v1
        vopc 09c 11 v_cmpx_gt_i16_e32 S0:v1 S1:v1
        vopc 09d 11 v_cmpx_ne_i16_e32 S0:v1 S1:v1
        vopc 09e 11 v_cmpx_ge_i16_e32 S0:v1 S1:v1
        vopc 09f 11 v_cmpx_class_f16_e32 S0:v1 S1:v1
        vopc 0a0 11 v_cmp_f_i64_e32 =vcc S0:v2 S1:v2
        vopc 0a1 11 v_cmp_lt_i64_e32 =vcc S0:v2 S1:v2
        vopc 0a2 11 v_cmp_eq_i64_e32 =vcc S0:v2 S1:v2
        vopc 0a3 11 v_cmp_le_i64_e32 =vcc S0:v2 S1:v2
        vopc 0a4 11 v_cmp_gt_i64_e32 =vcc S0:v2 S1:v2
        vopc 0a5 11 v_cmp_ne_i64_e32 =vcc S0:v2 S1:v2
        vopc 0a6 11 v_cmp_ge_i64_e32 =vcc S0:v2 S1:v2
        vopc 0a7 11 v_cmp_t_i64_e32 =vcc S0:v2 S1:v2
        vopc 0a8 11 v_cmp_class_f64_e32 =vcc S0:v2 S1:v1
        vopc 0a9 11 v_cmp_lt_u16_e32 =vcc S0:v1 S1:v1
        vopc 0aa 11 v_cmp_eq_u16_e32 =vcc S0:v1 S1:v1
        vopc 0ab 11 v_cmp_le_u16_e32 =vcc S0:v1 S1:v1
        vopc 0ac 11 v_cmp_gt_u16_e32 =vcc S0:v1 S1:v1
        vopc 0ad 11 v_cmp_ne_u16_e32 =vcc S0:v1 S1:v1
        vopc 0ae 11 v_cmp_ge_u16_e32 =vcc S0:v1 S1:v1
        vopc 0b0 11 v_cmpx_f_i64_e32 S0:v2 S1:v2
        vopc 0b1 11 v_cmpx_lt_i64_e32 S0:v2 S1:v2
        vopc 0b2 11 v_cmpx_eq_i64_e32 S0:v2 S1:v2
        vopc 0b3 11 v_cmpx_le_i64_e32 S0:v2 S1:v2
        vopc 0b4 11 v_cmpx_gt_i64_e32 S0:v2 S1:v2
        vopc 0b5 11 v_cmpx_ne_i64_e32 S0:v2 S1:v2
        vopc 0b6 11 v_cmpx_ge_i64_e32 S0:v2 S1:v2
        vopc 0b7 11 v_cmpx_t_i64_e32 S0:v2 S1:v2
        vopc 0b8 11 v_cmpx_class_f64_e32 S0:v2 S1:v1
        vopc 0b9 11 v_cmpx_lt_u16_e32 S0:v1 S1:v1
        vopc 0ba 11 v_cmpx_eq_u16_e32 S0:v1 S1:v1
        vopc 0bb 11 v_cmpx_le_u16_e32 S0:v1 S1:v1
        vopc 0bc 11 v_cmpx_gt_u16_e32 S0:v1 S1:v1
        vopc 0bd 11 v_cmpx_ne_u16_e32 S0:v1 S1:v1
        vopc 0be 11 v_cmpx_ge_u16_e32 S0:v1 S1:v1
        vopc 0c0 11 v_cmp_f_u32_e32 =vcc S0:v1 S1:v1
        vopc 0c1 11 v_cmp_lt_u32_e32 =vcc S0:v1 S1:v1
        vopc 0c2 11 v_cmp_eq_u32_e32 =vcc S0:v1 S1:v1
        vopc 0c3 11 v_cmp_le_u32_e32 =vcc S0:v1 S1:v1
        vopc 0c4 11 v_cmp_gt_u32_e32 =vcc S0:v1 S1:v1
        vopc 0c5 11 v_cmp_ne_u32_e32 =vcc S0:v1 S1:v1
        vopc 0c6 11 v_cmp_ge_u32_e32 =vcc S0:v1 S1:v1
        vopc 0c7 11 v_cmp_t_u32_e32 =vcc S0:v1 S1:v1
        vopc 0c8 11 v_cmp_f_f16_e32 =vcc S0:v1 S1:v1
        vopc 0c9 11 v_cmp_lt_f16_e32 =vcc S0:v1 S1:v1
        vopc 0ca 11 v_cmp_eq_f16_e32 =vcc S0:v1 S1:v1
        vopc 0cb 11 v_cmp_le_f16_e32 =vcc S0:v1 S1:v1
        vopc 0cc 11 v_cmp_gt_f16_e32 =vcc S0:v1 S1:v1
        vopc 0cd 11 v_cmp_lg_f16_e32 =vcc S0:v1 S1:v1
        vopc 0ce 11 v_cmp_ge_f16_e32 =vcc S0:v1 S1:v1
        vopc 0cf 11 v_cmp_o_f16_e32 =vcc S0:v1 S1:v1
        vopc 0d0 11 v_cmpx_f_u32_e32 S0:v1 S1:v1
        vopc 0d1 11 v_cmpx_lt_u32_e32 S0:v1 S1:v1
        vopc 0d2 11 v_cmpx_eq_u32_e32 S0:v1 S1:v1
        vopc 0d3 11 v_cmpx_le_u32_e32 S0:v1 S1:v1
        vopc 0d4 11 v_cmpx_gt_u32_e32 S0:v1 S1:v1
        vopc 0d5 11 v_cmpx_ne_u32_e32 S0:v1 S1:v1
        vopc 0d6 11 v_cmpx_ge_u32_e32 S0:v1 S1:v1
        vopc 0d7 11 v_cmpx_t_u32_e32 S0:v1 S1:v1
        vopc 0d8 11 v_cmpx_f_f16_e32 S0:v1 S1:v1
        vopc 0d9 11 v_cmpx_lt_f16_e32 S0:v1 S1:v1
        vopc 0da 11 v_cmpx_eq_f16_e32 S0:v1 S1:v1
        vopc 0db 11 v_cmpx_le_f16_e32 S0:v1 S1:v1
        vopc 0dc 11 v_cmpx_gt_f16_e32 S0:v1 S1:v1
        vopc 0dd 11 v_cmpx_lg_f16_e32 S0:v1 S1:v1
        vopc 0de 11 v_cmpx_ge_f16_e32 S0:v1 S1:v1
        vopc 0df 11 v_cmpx_o_f16_e32 S0:v1 S1:v1
        vopc 0e0 11 v_cmp_f_u64_e32 =vcc S0:v2 S1:v2
        vopc 0e1 11 v_cmp_lt_u64_e32 =vcc S0:v2 S1:v2
        vopc 0e2 11 v_cmp_eq_u64_e32 =vcc S0:v2 S1:v2
        vopc 0e3 11 v_cmp_le_u64_e32 =vcc S0:v2 S1:v2
        vopc 0e4 11 v_cmp_gt_u64_e32 =vcc S0:v2 S1:v2
        vopc 0e5 11 v_cmp_ne_u64_e32 =vcc S0:v2 S1:v2
        vopc 0e6 11 v_cmp_ge_u64_e32 =vcc S0:v2 S1:v2
        vopc 0e7 11 v_cmp_t_u64_e32 =vcc S0:v2 S1:v2
        vopc 0e8 11 v_cmp_u_f16_e32 =vcc S0:v1 S1:v1
        vopc 0e9 11 v_cmp_nge_f16_e32 =vcc S0:v1 S1:v1
        vopc 0ea 11 v_cmp_nlg_f16_e32 =vcc S0:v1 S1:v1
        vopc 0eb 11 v_cmp_ngt_f16_e32 =vcc S0:v1 S1:v1
        vopc 0ec 11 v_cmp_nle_f16_e32 =vcc S0:v1 S1:v1
        vopc 0ed 11 v_cmp_neq_f16_e32 =vcc S0:v1 S1:v1
        vopc 0ee 11 v_cmp_nlt_f16_e32 =vcc S0:v1 S1:v1
        vopc 0ef 11 v_cmp_tru_f16_e32 =vcc S0:v1 S1:v1
        vopc 0f0 11 v_cmpx_f_u64_e32 S0:v2 S1:v2
        vopc 0f1 11 v_cmpx_lt_u64_e32 S0:v2 S1:v2
        vopc 0f2 11 v_cmpx_eq_u64_e32 S0:v2 S1:v2
        vopc 0f3 11 v_cmpx_le_u64_e32 S0:v2 S1:v2
        vopc 0f4 11 v_cmpx_gt_u64_e32 S0:v2 S1:v2
        vopc 0f5 11 v_cmpx_ne_u64_e32 S0:v2 S1:v2
        vopc 0f6 11 v_cmpx_ge_u64_e32 S0:v2 S1:v2
        vopc 0f7 11 v_cmpx_t_u64_e32 S0:v2 S1:v2
        vopc 0f8 11 v_cmpx_u_f16_e32 S0:v1 S1:v1
        vopc 0f9 11 v_cmpx_nge_f16_e32 S0:v1 S1:v1
        vopc 0fa 11 v_cmpx_nlg_f16_e32 S0:v1 S1:v1
        vopc 0fb 11 v_cmpx_ngt_f16_e32 S0:v1 S1:v1
        vopc 0fc 11 v_cmpx_nle_f16_e32 S0:v1 S1:v1
        vopc 0fd 11 v_cmpx_neq_f16_e32 S0:v1 S1:v1
        vopc 0fe 11 v_cmpx_nlt_f16_e32 S0:v1 S1:v1
        vopc 0ff 11 v_cmpx_tru_f16_e32 S0:v1 S1:v1
        vop3 000 11 v_cmp_f_f32_e64 D:s2 S0:v1 S1:v1
        vop3 001 11 v_cmp_lt_f32_e64 D:s2 S0:v1 S1:v1
        vop3 002 11 v_cmp_eq_f32_e64 D:s2 S0:v1 S1:v1
        vop3 003 11 v_cmp_le_f32_e64 D:s2 S0:v1 S1:v1
        vop3 004 11 v_cmp_gt_f32_e64 D:s2 S0:v1 S1:v1
        vop3 005 11 v_cmp_lg_f32_e64 D:s2 S0:v1 S1:v1
        vop3 006 11 v_cmp_ge_f32_e64 D:s2 S0:v1 S1:v1
        vop3 007 11 v_cmp_o_f32_e64 D:s2 S0:v1 S1:v1
        vop3 008 11 v_cmp_u_f32_e64 D:s2 S0:v1 S1:v1
        vop3 009 11 v_cmp_nge_f32_e64 D:s2 S0:v1 S1:v1
        vop3 00a 11 v_cmp_nlg_f32_e64 D:s2 S0:v1 S1:v1
        vop3 00b 11 v_cmp_ngt_f32_e64 D:s2 S0:v1 S1:v1
        vop3 00c 11 v_cmp_nle_f32_e64 D:s2 S0:v1 S1:v1
        vop3 00d 11 v_cmp_neq_f32_e64 D:s2 S0:v1 S1:v1
        vop3 00e 11 v_cmp_nlt_f32_e64 D:s2 S0:v1 S1:v1
        vop3 00f 11 v_cmp_tru_f32_e64 D:s2 S0:v1 S1:v1
        vop3 010 11 v_cmpx_f_f32_e64 S0:v1 S1:v1
        vop3 011 11 v_cmpx_lt_f32_e64 S0:v1 S1:v1
        vop3 012 11 v_cmpx_eq_f32_e64 S0:v1 S1:v1
        vop3 013 11 v_cmpx_le_f32_e64 S0:v1 S1:v1
        vop3 014 11 v_cmpx_gt_f32_e64 S0:v1 S1:v1
        vop3 015 11 v_cmpx_lg_f32_e64 S0:v1 S1:v1
        vop3 016 11 v_cmpx_ge_f32_e64 S0:v1 S1:v1
        vop3 017 11 v_cmpx_o_f32_e64 S0:v1 S1:v1
        vop3 018 11 v_cmpx_u_f32_e64 S0:v1 S1:v1
        vop3 019 11 v_cmpx_nge_f32_e64 S0:v1 S1:v1
        vop3 01a 11 v_cmpx_nlg_f32_e64 S0:v1 S1:v1
        vop3 01b 11 v_cmpx_ngt_f32_e64 S0:v1 S1:v1
        vop3 01c 11 v_cmpx_nle_f32_e64 S0:v1 S1:v1
        vop3 01d 11 v_cmpx_neq_f32_e64 S0:v1 S1:v1
        vop3 01e 11 v_cmpx_nlt_f32_e64 S0:v1 S1:v1
        vop3 01f 11 v_cmpx_tru_f32_e64 S0:v1 S1:v1
        vop3 020 11 v_cmp_f_f64_e64 D:s2 S0:v2 S1:v2
        vop3 021 11 v_cmp_lt_f64_e64 D:s2 S0:v2 S1:v2
        vop3 022 11 v_cmp_eq_f64_e64 D:s2 S0:v2 S1:v2
        vop3 023 11 v_cmp_le_f64_e64 D:s2 S0:v2 S1:v2
        vop3 024 11 v_cmp_gt_f64_e64 D:s2 S0:v2 S1:v2
        vop3 025 11 v_cmp_lg_f64_e64 D:s2 S0:v2 S1:v2
        vop3 026 11 v_cmp_ge_f64_e64 D:s2 S0:v2 S1:v2
        vop3 027 11 v_cmp_o_f64_e64 D:s2 S0:v2 S1:v2
        vop3 028 11 v_cmp_u_f64_e64 D:s2 S0:v2 S1:v2
        vop3 029 11 v_cmp_nge_f64_e64 D:s2 S0:v2 S1:v2
        vop3 02a 11 v_cmp_nlg_f64_e64 D:s2 S0:v2 S1:v2
        vop3 02b 11 v_cmp_ngt_f64_e64 D:s2 S0:v2 S1:v2
        vop3 02c 11 v_cmp_nle_f64_e64 D:s2 S0:v2 S1:v2
        vop3 02d 11 v_cmp_neq_f64_e64 D:s2 S0:v2 S1:v2
        vop3 02e 11 v_cmp_nlt_f64_e64 D:s2 S0:v2 S1:v2
        vop3 02f 11 v_cmp_tru_f64_e64 D:s2 S0:v2 S1:v2
        vop3 030 11 v_cmpx_f_f64_e64 S0:v2 S1:v2
        vop3 031 11 v_cmpx_lt_f64_e64 S0:v2 S1:v2
        vop3 032 11 v_cmpx_eq_f64_e64 S0:v2 S1:v2
        vop3 033 11 v_cmpx_le_f64_e64 S0:v2 S1:v2
        vop3 034 11 v_cmpx_gt_f64_e64 S0:v2 S1:v2
        vop3 035 11 v_cmpx_lg_f64_e64 S0:v2 S1:v2
        vop3 036 11 v_cmpx_ge_f64_e64 S0:v2 S1:v2
        vop3 037 11 v_cmpx_o_f64_e64 S0:v2 S1:v2
        vop3 038 11 v_cmpx_u_f64_e64 S0:v2 S1:v2
        vop3 039 11 v_cmpx_nge_f64_e64 S0:v2 S1:v2
        vop3 03a 11 v_cmpx_nlg_f64_e64 S0:v2 S1:v2
        vop3 03b 11 v_cmpx_ngt_f64_e64 S0:v2 S1:v2
        vop3 03c 11 v_cmpx_nle_f64_e64 S0:v2 S1:v2
        vop3 03d 11 v_cmpx_neq_f64_e64 S0:v2 S1:v2
        vop3 03e 11 v_cmpx_nlt_f64_e64 S0:v2 S1:v2
        vop3 03f 11 v_cmpx_tru_f64_e64 S0:v2 S1:v2
        vop3 080 11 v_cmp_f_i32_e64 D:s2 S0:v1 S1:v1
        vop3 081 11 v_cmp_lt_i32_e64 D:s2 S0:v1 S1:v1
        vop3 082 11 v_cmp_eq_i32_e64 D:s2 S0:v1 S1:v1
        vop3 083 11 v_cmp_le_i32_e64 D:s2 S0:v1 S1:v1
        vop3 084 11 v_cmp_gt_i32_e64 D:s2 S0:v1 S1:v1
        vop3 085 11 v_cmp_ne_i32_e64 D:s2 S0:v1 S1:v1
        vop3 086 11 v_cmp_ge_i32_e64 D:s2 S0:v1 S1:v1
        vop3 087 11 v_cmp_t_i32_e64 D:s2 S0:v1 S1:v1
        vop3 088 11 v_cmp_class_f32_e64 D:s2 S0:v1 S1:v1
        vop3 089 11 v_cmp_lt_i16_e64 D:s2 S0:v1 S1:v1
        vop3 08a 11 v_cmp_eq_i16_e64 D:s2 S0:v1 S1:v1
        vop3 08b 11 v_cmp_le_i16_e64 D:s2 S0:v1 S1:v1
        vop3 08c 11 v_cmp_gt_i16_e64 D:s2 S0:v1 S1:v1
        vop3 08d 11 v_cmp_ne_i16_e64 D:s2 S0:v1 S1:v1
        vop3 08e 11 v_cmp_ge_i16_e64 D:s2 S0:v1 S1:v1
        vop3 08f 11 v_cmp_class_f16_e64 D:s2 S0:v1 S1:v1
        vop3 090 11 v_cmpx_f_i32_e64 S0:v1 S1:v1
        vop3 091 11 v_cmpx_lt_i32_e64 S0:v1 S1:v1
        vop3 092 11 v_cmpx_eq_i32_e64 S0:v1 S1:v1
        vop3 093 11 v_cmpx_le_i32_e64 S0:v1 S1:v1
        vop3 094 11 v_cmpx_gt_i32_e64 S0:v1 S1:v1
        vop3 095 11 v_cmpx_ne_i32_e64 S0:v1 S1:v1
        vop3 096 11 v_cmpx_ge_i32_e64 S0:v1 S1:v1
        vop3 097 11 v_cmpx_t_i32_e64 S0:v1 S1:v1
        vop3 098 11 v_cmpx_class_f32_e64 S0:v1 S1:v1
        vop3 099 11 v_cmpx_lt_i16_e64 S0:v1 S1:v1
        vop3 09a 11 v_cmpx_eq_i16_e64 S0:v1 S1:v1
        vop3 09b 11 v_cmpx_le_i16_e64 S0:v1 S1:v1
        vop3 09c 11 v_cmpx_gt_i16_e64 S0:v1 S1:v1
        vop3 09d 11 v_cmpx_ne_i16_e64 S0:v1 S1:v1
        vop3 09e 11 v_cmpx_ge_i16_e64 S0:v1 S1:v1
        vop3 09f 11 v_cmpx_class_f16_e64 S0:v1 S1:v1
        vop3 0a0 11 v_cmp_f_i64_e64 D:s2 S0:v2 S1:v2
        vop3 0a1 11 v_cmp_lt_i64_e64 D:s2 S0:v2 S1:v2
        vop3 0a2 11 v_cmp_eq_i64_e64 D:s2 S0:v2 S1:v2
        vop3 0a3 11 v_cmp_le_i64_e64 D:s2 S0:v2 S1:v2
        vop3 0a4 11 v_cmp_gt_i64_e64 D:s2 S0:v2 S1:v2
        vop3 0a5 11 v_cmp_ne_i64_e64 D:s2 S0:v2 S1:v2
        vop3 0a6 11 v_cmp_ge_i64_e64 D:s2 S0:v2 S1:v2
        vop3 0a7 11 v_cmp_t_i64_e64 D:s2 S0:v2 S1:v2
        vop3 0a8 11 v_cmp_class_f64_e64 D:s2 S0:v2 S1:v1
        vop3 0a9 11 v_cmp_lt_u16_e64 D:s2 S0:v1 S1:v1
        vop3 0aa 11 v_cmp_eq_u16_e64 D:s2 S0:v1 S1:v1
        vop3 0ab 11 v_cmp_le_u16_e64 D:s2 S0:v1 S1:v1
        vop3 0ac 11 v_cmp_gt_u16_e64 D:s2 S0:v1 S1:v1
        vop3 0ad 11 v_cmp_ne_u16_e64 D:s2 S0:v1 S1:v1
        vop3 0ae 11 v_cmp_ge_u16_e64 D:s2 S0:v1 S1:v1
        vop3 0b0 11 v_cmpx_f_i64_e64 S0:v2 S1:v2
        vop3 0b1 11 v_cmpx_lt_i64_e64 S0:v2 S1:v2
        vop3 0b2 11 v_cmpx_eq_i64_e64 S0:v2 S1:v2
        vop3 0b3 11 v_cmpx_le_i64_e64 S0:v2 S1:v2
        vop3 0b4 11 v_cmpx_gt_i64_e64 S0:v2 S1:v2
        vop3 0b5 11 v_cmpx_ne_i64_e64 S0:v2 S1:v2
        vop3 0b6 11 v_cmpx_ge_i64_e64 S0:v2 S1:v2
        vop3 0b7 11 v_cmpx_t_i64_e64 S0:v2 S1:v2
        vop3 0b8 11 v_cmpx_class_f64_e64 S0:v2 S1:v1
        vop3 0b9 11 v_cmpx_lt_u16_e64 S0:v1 S1:v1
        vop3 0ba 11 v_cmpx_eq_u16_e64 S0:v1 S1:v1
        vop3 0bb 11 v_cmpx_le_u16_e64 S0:v1 S1:v1
        vop3 0bc 11 v_cmpx_gt_u16_e64 S0:v1 S1:v1
        vop3 0bd 11 v_cmpx_ne_u16_e64 S0:v1 S1:v1
        vop3 0be 11 v_cmpx_ge_u16_e64 S0:v1 S1:v1
        vop3 0c0 11 v_cmp_f_u32_e64 D:s2 S0:v1 S1:v1
        vop3 0c1 11 v_cmp_lt_u32_e64 D:s2 S0:v1 S1:v1
        vop3 0c2 11 v_cmp_eq_u32_e64 D:s2 S0:v1 S1:v1
        vop3 0c3 11 v_cmp_le_u32_e64 D:s2 S0:v1 S1:v1
        vop3 0c4 11 v_cmp_gt_u32_e64 D:s2 S0:v1 S1:v1
        vop3 0c5 11 v_cmp_ne_u32_e64 D:s2 S0:v1 S1:v1
        vop3 0c6 11 v_cmp_ge_u32_e64 D:s2 S0:v1 S1:v1
        vop3 0c7 11 v_cmp_t_u32_e64 D:s2 S0:v1 S1:v1
        vop3 0c8 11 v_cmp_f_f16_e64 D:s2 S0:v1 S1:v1
        vop3 0c9 11 v_cmp_lt_f16_e64 D:s2 S0:v1 S1:v1
        vop3 0ca 11 v_cmp_eq_f16_e64 D:s2 S0:v1 S1:v1
        vop3 0cb 11 v_cmp_le_f16_e64 D:s2 S0:v1 S1:v1
        vop3 0cc 11 v_cmp_gt_f16_e64 D:s2 S0:v1 S1:v1
        vop3 0cd 11 v_cmp_lg_f16_e64 D:s2 S0:v1 S1:v1
        vop3 0ce 11 v_cmp_ge_f16_e64 D:s2 S0:v1 S1:v1
        vop3 0cf 11 v_cmp_o_f16_e64 D:s2 S0:v1 S1:v1
        vop3 0d0 11 v_cmpx_f_u32_e64 S0:v1 S1:v1
        vop3 0d1 11 v_cmpx_lt_u32_e64 S0:v1 S1:v1
        vop3 0d2 11 v_cmpx_eq_u32_e64 S0:v1 S1:v1
        vop3 0d3 11 v_cmpx_le_u32_e64 S0:v1 S1:v1
        vop3 0d4 11 v_cmpx_gt_u32_e64 S0:v1 S1:v1
        vop3 0d5 11 v_cmpx_ne_u32_e64 S0:v1 S1:v1
        vop3 0d6 11 v_cmpx_ge_u32_e64 S0:v1 S1:v1
        vop3 0d7 11 v_cmpx_t_u32_e64 S0:v1 S1:v1
        vop3 0d8 11 v_cmpx_f_f16_e64 S0:v1 S1:v1
        vop3 0d9 11 v_cmpx_lt_f16_e64 S0:v1 S1:v1
        vop3 0da 11 v_cmpx_eq_f16_e64 S0:v1 S1:v1
        vop3 0db 11 v_cmpx_le_f16_e64 S0:v1 S1:v1
        vop3 0dc 11 v_cmpx_gt_f16_e64 S0:v1 S1:v1
        vop3 0dd 11 v_cmpx_lg_f16_e64 S0:v1 S1:v1
        vop3 0de 11 v_cmpx_ge_f16_e64 S0:v1 S1:v1
        vop3 0df 11 v_cmpx_o_f16_e64 S0:v1 S1:v1
        vop3 0e0 11 v_cmp_f_u64_e64 D:s2 S0:v2 S1:v2
        vop3 0e1 11 v_cmp_lt_u64_e64 D:s2 S0:v2 S1:v2
        vop3 0e2 11 v_cmp_eq_u64_e64 D:s2 S0:v2 S1:v2
        vop3 0e3 11 v_cmp_le_u64_e64 D:s2 S0:v2 S1:v2
        vop3 0e4 11 v_cmp_gt_u64_e64 D:s2 S0:v2 S1:v2
        vop3 0e5 11 v_cmp_ne_u64_e64 D:s2 S0:v2 S1:v2
        vop3 0e6 11 v_cmp_ge_u64_e64 D:s2 S0:v2 S1:v2
        vop3 0e7 11 v_cmp_t_u64_e64 D:s2 S0:v2 S1:v2
        vop3 0e8 11 v_cmp_u_f16_e64 D:s2 S0:v1 S1:v1
        vop3 0e9 11 v_cmp_nge_f16_e64 D:s2 S0:v1 S1:v1
        vop3 0ea 11 v_cmp_nlg_f16_e64 D:s2 S0:v1 S1:v1
        vop3 0eb 11 v_cmp_ngt_f16_e64 D:s2 S0:v1 S1:v1
        vop3 0ec 11 v_cmp_nle_f16_e64 D:s2 S0:v1 S1:v1
        vop3 0ed 11 v_cmp_neq_f16_e64 D:s2 S0:v1 S1:v1
        vop3 0ee 11 v_cmp_nlt_f16_e64 D:s2 S0:v1 S1:v1
        vop3 0ef 11 v_cmp_tru_f16_e64 D:s2 S0:v1 S1:v1
        vop3 0f0 11 v_cmpx_f_u64_e64 S0:v2 S1:v2
        vop3 0f1 11 v_cmpx_lt_u64_e64 S0:v2 S1:v2
        vop3 0f2 11 v_cmpx_eq_u64_e64 S0:v2 S1:v2
        vop3 0f3 11 v_cmpx_le_u64_e64 S0:v2 S1:v2
        vop3 0f4 11 v_cmpx_gt_u64_e64 S0:v2 S1:v2
        vop3 0f5 11 v_cmpx_ne_u64_e64 S0:v2 S1:v2
        vop3 0f6 11 v_cmpx_ge_u64_e64 S0:v2 S1:v2
        vop3 0f7 11 v_cmpx_t_u64_e64 S0:v2 S1:v2
        vop3 0f8 11 v_cmpx_u_f16_e64 S0:v1 S1:v1
        vop3 0f9 11 v_cmpx_nge_f16_e64 S0:v1 S1:v1
        vop3 0fa 11 v_cmpx_nlg_f16_e64 S0:v1 S1:v1
        vop3 0fb 11 v_cmpx_ngt_f16_e64 S0:v1 S1:v1
        vop3 0fc 11 v_cmpx_nle_f16_e64 S0:v1 S1:v1
        vop3 0fd 11 v_cmpx_neq_f16_e64 S0:v1 S1:v1
        vop3 0fe 11 v_cmpx_nlt_f16_e64 S0:v1 S1:v1
        vop3 0ff 11 v_cmpx_tru_f16_e64 S0:v1 S1:v1
        vop3 101 11 v_cndmask_b32_e64 D:v1 S0:v1 S1:v1 S2:s2
        vop3 103 11 v_add_f32_e64 D:v1 S0:v1 S1:v1
        vop3 104 11 v_sub_f32_e64 D:v1 S0:v1 S1:v1
        vop3 105 11 v_subrev_f32_e64 D:v1 S0:v1 S1:v1
        vop3 106 01 v_fmac_legacy_f32_e64 D:v1 S0:v1 S1:v1
        vop3 106 10 v_mac_legacy_f32_e64 D:v1 S0:v1 S1:v1
        vop3 107 11 v_mul_legacy_f32_e64 D:v1 S0:v1 S1:v1
        vop3 108 11 v_mul_f32_e64 D:v1 S0:v1 S1:v1
        vop3 109 11 v_mul_i32_i24_e64 D:v1 S0:v1 S1:v1
        vop3 10a 11 v_mul_hi_i32_i24_e64 D:v1 S0:v1 S1:v1
        vop3 10b 11 v_mul_u32_u24_e64 D:v1 S0:v1 S1:v1
        vop3 10c 11 v_mul_hi_u32_u24_e64 D:v1 S0:v1 S1:v1
        vop3 10f 11 v_min_f32_e64 D:v1 S0:v1 S1:v1
        vop3 110 11 v_max_f32_e64 D:v1 S0:v1 S1:v1
        vop3 111 11 v_min_i32_e64 D:v1 S0:v1 S1:v1
        vop3 112 11 v_max_i32_e64 D:v1 S0:v1 S1:v1
        vop3 113 11 v_min_u32_e64 D:v1 S0:v1 S1:v1
        vop3 114 11 v_max_u32_e64 D:v1 S0:v1 S1:v1
        vop3 116 11 v_lshrrev_b32_e64 D:v1 S0:v1 S1:v1
        vop3 118 11 v_ashrrev_i32_e64 D:v1 S0:v1 S1:v1
        vop3 11a 11 v_lshlrev_b32_e64 D:v1 S0:v1 S1:v1
        vop3 11b 11 v_and_b32_e64 D:v1 S0:v1 S1:v1
        vop3 11c 11 v_or_b32_e64 D:v1 S0:v1 S1:v1
        vop3 11d 11 v_xor_b32_e64 D:v1 S0:v1 S1:v1
        vop3 11e 11 v_xnor_b32_e64 D:v1 S0:v1 S1:v1
        vop3 11f 10 v_mac_f32_e64 D:v1 S0:v1 S1:v1
        vop3 125 11 v_add_nc_u32_e64 D:v1 S0:v1 S1:v1
        vop3 126 11 v_sub_nc_u32_e64 D:v1 S0:v1 S1:v1
        vop3 127 11 v_subrev_nc_u32_e64 D:v1 S0:v1 S1:v1
        vop3 128 11 v_add_co_ci_u32_e64 D:v1 SD:s2 S0:v1 S1:v1 S2:s2
        vop3 129 11 v_sub_co_ci_u32_e64 D:v1 SD:s2 S0:v1 S1:v1 S2:s2
        vop3 12a 11 v_subrev_co_ci_u32_e64 D:v1 SD:s2 S0:v1 S1:v1 S2:s2
        vop3 12b 11 v_fmac_f32_e64 D:v1 S0:v1 S1:v1
        vop3 12f 11 v_cvt_pkrtz_f16_f32_e64 D:v1 S0:v1 S1:v1
        vop3 132 11 v_add_f16_e64 D:v1 S0:v1 S1:v1
        vop3 133 11 v_sub_f16_e64 D:v1 S0:v1 S1:v1
        vop3 134 11 v_subrev_f16_e64 D:v1 S0:v1 S1:v1
        vop3 135 11 v_mul_f16_e64 D:v1 S0:v1 S1:v1
        vop3 136 11 v_fmac_f16_e64 D:v1 S0:v1 S1:v1
        vop3 139 11 v_max_f16_e64 D:v1 S0:v1 S1:v1
        vop3 13a 11 v_min_f16_e64 D:v1 S0:v1 S1:v1
        vop3 13b 11 v_ldexp_f16_e64 D:v1 S0:v1 S1:v1
        vop3 140 01 v_fma_legacy_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 140 10 v_mad_legacy_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 141 10 v_mad_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 142 11 v_mad_i32_i24 D:v1 S0:v1 S1:v1 S2:v1
        vop3 143 11 v_mad_u32_u24 D:v1 S0:v1 S1:v1 S2:v1
        vop3 144 11 v_cubeid_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 145 11 v_cubesc_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 146 11 v_cubetc_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 147 11 v_cubema_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 148 11 v_bfe_u32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 149 11 v_bfe_i32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 14a 11 v_bfi_b32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 14b 11 v_fma_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 14c 11 v_fma_f64 D:v2 S0:v2 S1:v2 S2:v2
        vop3 14d 11 v_lerp_u8 D:v1 S0:v1 S1:v1 S2:v1
        vop3 14e 11 v_alignbit_b32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 14f 11 v_alignbyte_b32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 150 11 v_mullit_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 151 11 v_min3_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 152 11 v_min3_i32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 153 11 v_min3_u32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 154 11 v_max3_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 155 11 v_max3_i32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 156 11 v_max3_u32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 157 11 v_med3_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 158 11 v_med3_i32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 159 11 v_med3_u32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 15a 11 v_sad_u8 D:v1 S0:v1 S1:v1 S2:v1
        vop3 15b 11 v_sad_hi_u8 D:v1 S0:v1 S1:v1 S2:v1
        vop3 15c 11 v_sad_u16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 15d 11 v_sad_u32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 15e 11 v_cvt_pk_u8_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 15f 11 v_div_fixup_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 160 11 v_div_fixup_f64 D:v2 S0:v2 S1:v2 S2:v2
        vop3 164 11 v_add_f64 D:v2 S0:v2 S1:v2
        vop3 165 11 v_mul_f64 D:v2 S0:v2 S1:v2
        vop3 166 11 v_min_f64 D:v2 S0:v2 S1:v2
        vop3 167 11 v_max_f64 D:v2 S0:v2 S1:v2
        vop3 168 11 v_ldexp_f64 D:v2 S0:v2 S1:v1
        vop3 169 11 v_mul_lo_u32 D:v1 S0:v1 S1:v1
        vop3 16a 11 v_mul_hi_u32 D:v1 S0:v1 S1:v1
        vop3 16b 10 v_mul_lo_i32 D:v1 S0:v1 S1:v1
        vop3 16c 11 v_mul_hi_i32 D:v1 S0:v1 S1:v1
        vop3 16d 11 v_div_scale_f32 D:v1 SD:s2 S0:v1 S1:v1 S2:v1
        vop3 16e 11 v_div_scale_f64 D:v2 SD:s2 S0:v2 S1:v2 S2:v2
        vop3 16f 11 v_div_fmas_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 170 11 v_div_fmas_f64 D:v2 S0:v2 S1:v2 S2:v2
        vop3 171 11 v_msad_u8 D:v1 S0:v1 S1:v1 S2:v1
        vop3 172 11 v_qsad_pk_u16_u8 D:v2 S0:v2 S1:v1 S2:v2
        vop3 173 11 v_mqsad_pk_u16_u8 D:v2 S0:v2 S1:v1 S2:v2
        vop3 174 11 v_trig_preop_f64 D:v2 S0:v2 S1:v1
        vop3 175 11 v_mqsad_u32_u8 D:v4 S0:v2 S1:v1 S2:v4
        vop3 176 11 v_mad_u64_u32 D:v2 SD:s2 S0:v1 S1:v1 S2:v2
        vop3 177 11 v_mad_i64_i32 D:v2 SD:s2 S0:v1 S1:v1 S2:v2
        vop3 178 11 v_xor3_b32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 180 11 v_nop -
        vop3 181 11 v_mov_b32_e64 D:v1 S0:v1
        vop3 183 11 v_cvt_i32_f64_e64 D:v1 S0:v2
        vop3 184 11 v_cvt_f64_i32_e64 D:v2 S0:v1
        vop3 185 11 v_cvt_f32_i32_e64 D:v1 S0:v1
        vop3 186 11 v_cvt_f32_u32_e64 D:v1 S0:v1
        vop3 187 11 v_cvt_u32_f32_e64 D:v1 S0:v1
        vop3 188 11 v_cvt_i32_f32_e64 D:v1 S0:v1
        vop3 18a 11 v_cvt_f16_f32_e64 D:v1 S0:v1
        vop3 18b 11 v_cvt_f32_f16_e64 D:v1 S0:v1
        vop3 18c 11 v_cvt_rpi_i32_f32_e64 D:v1 S0:v1
        vop3 18d 11 v_cvt_flr_i32_f32_e64 D:v1 S0:v1
        vop3 18e 11 v_cvt_off_f32_i4_e64 D:v1 S0:v1
        vop3 18f 11 v_cvt_f32_f64_e64 D:v1 S0:v2
        vop3 190 11 v_cvt_f64_f32_e64 D:v2 S0:v1
        vop3 191 11 v_cvt_f32_ubyte0_e64 D:v1 S0:v1
        vop3 192 11 v_cvt_f32_ubyte1_e64 D:v1 S0:v1
        vop3 193 11 v_cvt_f32_ubyte2_e64 D:v1 S0:v1
        vop3 194 11 v_cvt_f32_ubyte3_e64 D:v1 S0:v1
        vop3 195 11 v_cvt_u32_f64_e64 D:v1 S0:v2
        vop3 196 11 v_cvt_f64_u32_e64 D:v2 S0:v1
        vop3 197 11 v_trunc_f64_e64 D:v2 S0:v2
        vop3 198 11 v_ceil_f64_e64 D:v2 S0:v2
        vop3 199 11 v_rndne_f64_e64 D:v2 S0:v2
        vop3 19a 11 v_floor_f64_e64 D:v2 S0:v2
        vop3 19b 11 v_pipeflush -
        vop3 1a0 11 v_fract_f32_e64 D:v1 S0:v1
        vop3 1a1 11 v_trunc_f32_e64 D:v1 S0:v1
        vop3 1a2 11 v_ceil_f32_e64 D:v1 S0:v1
        vop3 1a3 11 v_rndne_f32_e64 D:v1 S0:v1
        vop3 1a4 11 v_floor_f32_e64 D:v1 S0:v1
        vop3 1a5 11 v_exp_f32_e64 D:v1 S0:v1
        vop3 1a7 11 v_log_f32_e64 D:v1 S0:v1
        vop3 1aa 11 v_rcp_f32_e64 D:v1 S0:v1
        vop3 1ab 11 v_rcp_iflag_f32_e64 D:v1 S0:v1
        vop3 1ae 11 v_rsq_f32_e64 D:v1 S0:v1
        vop3 1af 11 v_rcp_f64_e64 D:v2 S0:v2
        vop3 1b1 11 v_rsq_f64_e64 D:v2 S0:v2
        vop3 1b3 11 v_sqrt_f32_e64 D:v1 S0:v1
        vop3 1b4 11 v_sqrt_f64_e64 D:v2 S0:v2
        vop3 1b5 11 v_sin_f32_e64 D:v1 S0:v1
        vop3 1b6 11 v_cos_f32_e64 D:v1 S0:v1
        vop3 1b7 11 v_not_b32_e64 D:v1 S0:v1
        vop3 1b8 11 v_bfrev_b32_e64 D:v1 S0:v1
        vop3 1b9 11 v_ffbh_u32_e64 D:v1 S0:v1
        vop3 1ba 11 v_ffbl_b32_e64 D:v1 S0:v1
        vop3 1bb 11 v_ffbh_i32_e64 D:v1 S0:v1
        vop3 1bc 11 v_frexp_exp_i32_f64_e64 D:v1 S0:v2
        vop3 1bd 11 v_frexp_mant_f64_e64 D:v2 S0:v2
        vop3 1be 11 v_fract_f64_e64 D:v2 S0:v2
        vop3 1bf 11 v_frexp_exp_i32_f32_e64 D:v1 S0:v1
        vop3 1c0 11 v_frexp_mant_f32_e64 D:v1 S0:v1
        vop3 1c1 11 v_clrexcp -
        vop3 1c2 11 v_movreld_b32_e64 D:v1 S0:v1
        vop3 1c3 11 v_movrels_b32_e64 D:v1 S0:v1
        vop3 1c4 11 v_movrelsd_b32_e64 D:v1 S0:v1
        vop3 1c8 11 v_movrelsd_2_b32_e64 D:v1 S0:v1
        vop3 1d0 11 v_cvt_f16_u16_e64 D:v1 S0:v1
        vop3 1d1 11 v_cvt_f16_i16_e64 D:v1 S0:v1
        vop3 1d2 11 v_cvt_u16_f16_e64 D:v1 S0:v1
        vop3 1d3 11 v_cvt_i16_f16_e64 D:v1 S0:v1
        vop3 1d4 11 v_rcp_f16_e64 D:v1 S0:v1
        vop3 1d5 11 v_sqrt_f16_e64 D:v1 S0:v1
        vop3 1d6 11 v_rsq_f16_e64 D:v1 S0:v1
        vop3 1d7 11 v_log_f16_e64 D:v1 S0:v1
        vop3 1d8 11 v_exp_f16_e64 D:v1 S0:v1
        vop3 1d9 11 v_frexp_mant_f16_e64 D:v1 S0:v1
        vop3 1da 11 v_frexp_exp_i16_f16_e64 D:v1 S0:v1
        vop3 1db 11 v_floor_f16_e64 D:v1 S0:v1
        vop3 1dc 11 v_ceil_f16_e64 D:v1 S0:v1
        vop3 1dd 11 v_trunc_f16_e64 D:v1 S0:v1
        vop3 1de 11 v_rndne_f16_e64 D:v1 S0:v1
        vop3 1df 11 v_fract_f16_e64 D:v1 S0:v1
        vop3 1e0 11 v_sin_f16_e64 D:v1 S0:v1
        vop3 1e1 11 v_cos_f16_e64 D:v1 S0:v1
        vop3 1e2 11 v_sat_pk_u8_i16_e64 D:v1 S0:v1
        vop3 1e3 11 v_cvt_norm_i16_f16_e64 D:v1 S0:v1
        vop3 1e4 11 v_cvt_norm_u16_f16_e64 D:v1 S0:v1
        vop3 2ff 11 v_lshlrev_b64 D:v2 S0:v1 S1:v2
        vop3 300 11 v_lshrrev_b64 D:v2 S0:v1 S1:v2
        vop3 301 11 v_ashrrev_i64 D:v2 S0:v1 S1:v2
        vop3 303 11 v_add_nc_u16 D:v1 S0:v1 S1:v1
        vop3 304 11 v_sub_nc_u16 D:v1 S0:v1 S1:v1
        vop3 305 11 v_mul_lo_u16 D:v1 S0:v1 S1:v1
        vop3 307 11 v_lshrrev_b16 D:v1 S0:v1 S1:v1
        vop3 308 11 v_ashrrev_i16 D:v1 S0:v1 S1:v1
        vop3 309 11 v_max_u16 D:v1 S0:v1 S1:v1
        vop3 30a 11 v_max_i16 D:v1 S0:v1 S1:v1
        vop3 30b 11 v_min_u16 D:v1 S0:v1 S1:v1
        vop3 30c 11 v_min_i16 D:v1 S0:v1 S1:v1
        vop3 30d 11 v_add_nc_i16 D:v1 S0:v1 S1:v1
        vop3 30e 11 v_sub_nc_i16 D:v1 S0:v1 S1:v1
        vop3 30f 11 v_add_co_u32 D:v1 SD:s2 S0:v1 S1:v1
        vop3 310 11 v_sub_co_u32 D:v1 SD:s2 S0:v1 S1:v1
        vop3 311 11 v_pack_b32_f16 D:v1 S0:v1 S1:v1
        vop3 312 11 v_cvt_pknorm_i16_f16 D:v1 S0:v1 S1:v1
        vop3 313 11 v_cvt_pknorm_u16_f16 D:v1 S0:v1 S1:v1
        vop3 314 11 v_lshlrev_b16 D:v1 S0:v1 S1:v1
        vop3 319 11 v_subrev_co_u32 D:v1 SD:s2 S0:v1 S1:v1
        vop3 340 11 v_mad_u16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 344 11 v_perm_b32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 345 11 v_xad_u32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 346 11 v_lshl_add_u32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 347 11 v_add_lshl_u32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 34b 11 v_fma_f16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 351 11 v_min3_f16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 352 11 v_min3_i16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 353 11 v_min3_u16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 354 11 v_max3_f16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 355 11 v_max3_i16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 356 11 v_max3_u16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 357 11 v_med3_f16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 358 11 v_med3_i16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 359 11 v_med3_u16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 35e 11 v_mad_i16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 35f 11 v_div_fixup_f16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 360 11 v_readlane_b32 D:s1 S0:v1 S1:s1
        vop3 361 11 v_writelane_b32 D:v1 S0:s1 S1:s1
        vop3 362 11 v_ldexp_f32 D:v1 S0:v1 S1:v1
        vop3 363 11 v_bfm_b32 D:v1 S0:v1 S1:v1
        vop3 364 11 v_bcnt_u32_b32 D:v1 S0:v1 S1:v1
        vop3 365 11 v_mbcnt_lo_u32_b32 D:v1 S0:v1 S1:v1
        vop3 366 11 v_mbcnt_hi_u32_b32 D:v1 S0:v1 S1:v1
        vop3 368 11 v_cvt_pknorm_i16_f32 D:v1 S0:v1 S1:v1
        vop3 369 11 v_cvt_pknorm_u16_f32 D:v1 S0:v1 S1:v1
        vop3 36a 11 v_cvt_pk_u16_u32 D:v1 S0:v1 S1:v1
        vop3 36b 11 v_cvt_pk_i16_i32 D:v1 S0:v1 S1:v1
        vop3 36d 11 v_add3_u32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 36f 11 v_lshl_or_b32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 371 11 v_and_or_b32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 372 11 v_or3_b32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 373 11 v_mad_u32_u16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 375 11 v_mad_i32_i16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 376 11 v_sub_nc_i32 D:v1 S0:v1 S1:v1
        vop3 377 11 v_permlane16_b32 D:v1 S0:v1 S1:s1 S2:s1
        vop3 378 11 v_permlanex16_b32 D:v1 S0:v1 S1:s1 S2:s1
        vop3 37f 11 v_add_nc_i32 D:v1 S0:v1 S1:v1
        vop3p 000 11 v_pk_mad_i16 D:v1 S0:v1 S1:v1 S2:v1
        vop3p 001 11 v_pk_mul_lo_u16 D:v1 S0:v1 S1:v1
        vop3p 002 11 v_pk_add_i16 D:v1 S0:v1 S1:v1
        vop3p 003 11 v_pk_sub_i16 D:v1 S0:v1 S1:v1
        vop3p 004 11 v_pk_lshlrev_b16 D:v1 S0:v1 S1:v1
        vop3p 005 11 v_pk_lshrrev_b16 D:v1 S0:v1 S1:v1
        vop3p 006 11 v_pk_ashrrev_i16 D:v1 S0:v1 S1:v1
        vop3p 007 11 v_pk_max_i16 D:v1 S0:v1 S1:v1
        vop3p 008 11 v_pk_min_i16 D:v1 S0:v1 S1:v1
        vop3p 009 11 v_pk_mad_u16 D:v1 S0:v1 S1:v1 S2:v1
        vop3p 00a 11 v_pk_add_u16 D:v1 S0:v1 S1:v1
        vop3p 00b 11 v_pk_sub_u16 D:v1 S0:v1 S1:v1
        vop3p 00c 11 v_pk_max_u16 D:v1 S0:v1 S1:v1
        vop3p 00d 11 v_pk_min_u16 D:v1 S0:v1 S1:v1
        vop3p 00e 11 v_pk_fma_f16 D:v1 S0:v1 S1:v1 S2:v1
        vop3p 00f 11 v_pk_add_f16 D:v1 S0:v1 S1:v1
        vop3p 010 11 v_pk_mul_f16 D:v1 S0:v1 S1:v1
        vop3p 011 11 v_pk_min_f16 D:v1 S0:v1 S1:v1
        vop3p 012 11 v_pk_max_f16 D:v1 S0:v1 S1:v1
        vop3p 013 01 v_dot2_f32_f16 D:v1 S0:v1 S1:v1 S2:v1
        vop3p 014 01 v_dot2_i32_i16 D:v1 S0:v1 S1:v1 S2:v1
        vop3p 015 01 v_dot2_u32_u16 D:v1 S0:v1 S1:v1 S2:v1
        vop3p 016 01 v_dot4_i32_i8 D:v1 S0:v1 S1:v1 S2:v1
        vop3p 017 01 v_dot4_u32_u8 D:v1 S0:v1 S1:v1 S2:v1
        vop3p 018 01 v_dot8_i32_i4 D:v1 S0:v1 S1:v1 S2:v1
        vop3p 019 01 v_dot8_u32_u4 D:v1 S0:v1 S1:v1 S2:v1
        vop3p 020 11 v_fma_mix_f32 D:v1 S0:v1 S1:v1 S2:v1 mix
        vop3p 021 11 v_fma_mixlo_f16 D:v1 S0:v1 S1:v1 S2:v1 mix
        vop3p 022 11 v_fma_mixhi_f16 D:v1 S0:v1 S1:v1 S2:v1 mix
        ds 000 11 ds_add_u32 A:v1 S0:v1 O:16
        ds 001 11 ds_sub_u32 A:v1 S0:v1 O:16
        ds 002 11 ds_rsub_u32 A:v1 S0:v1 O:16
        ds 003 11 ds_inc_u32 A:v1 S0:v1 O:16
        ds 004 11 ds_dec_u32 A:v1 S0:v1 O:16
        ds 005 11 ds_min_i32 A:v1 S0:v1 O:16
        ds 006 11 ds_max_i32 A:v1 S0:v1 O:16
        ds 007 11 ds_min_u32 A:v1 S0:v1 O:16
        ds 008 11 ds_max_u32 A:v1 S0:v1 O:16
        ds 009 11 ds_and_b32 A:v1 S0:v1 O:16
        ds 00a 11 ds_or_b32 A:v1 S0:v1 O:16
        ds 00b 11 ds_xor_b32 A:v1 S0:v1 O:16
        ds 00c 11 ds_mskor_b32 A:v1 S0:v1 S1:v1 O:16
        ds 00d 11 ds_write_b32 A:v1 S0:v1 O:16
        ds 00e 11 ds_write2_b32 A:v1 S0:v1 S1:v1 O:8x2
        ds 00f 11 ds_write2st64_b32 A:v1 S0:v1 S1:v1 O:8x2
        ds 010 11 ds_cmpst_b32 A:v1 S0:v1 S1:v1 O:16
        ds 011 11 ds_cmpst_f32 A:v1 S0:v1 S1:v1 O:16
        ds 012 11 ds_min_f32 A:v1 S0:v1 O:16
        ds 013 11 ds_max_f32 A:v1 S0:v1 O:16
        ds 015 11 ds_add_f32 A:v1 S0:v1 O:16
        ds 01e 11 ds_write_b8 A:v1 S0:v1 O:16
        ds 01f 11 ds_write_b16 A:v1 S0:v1 O:16
        ds 020 11 ds_add_rtn_u32 D:v1 A:v1 S0:v1 O:16
        ds 021 11 ds_sub_rtn_u32 D:v1 A:v1 S0:v1 O:16
        ds 022 11 ds_rsub_rtn_u32 D:v1 A:v1 S0:v1 O:16
        ds 023 11 ds_inc_rtn_u32 D:v1 A:v1 S0:v1 O:16
        ds 024 11 ds_dec_rtn_u32 D:v1 A:v1 S0:v1 O:16
        ds 025 11 ds_min_rtn_i32 D:v1 A:v1 S0:v1 O:16
        ds 026 11 ds_max_rtn_i32 D:v1 A:v1 S0:v1 O:16
        ds 027 11 ds_min_rtn_u32 D:v1 A:v1 S0:v1 O:16
        ds 028 11 ds_max_rtn_u32 D:v1 A:v1 S0:v1 O:16
        ds 029 11 ds_and_rtn_b32 D:v1 A:v1 S0:v1 O:16
        ds 02a 11 ds_or_rtn_b32 D:v1 A:v1 S0:v1 O:16
        ds 02b 11 ds_xor_rtn_b32 D:v1 A:v1 S0:v1 O:16
        ds 02c 11 ds_mskor_rtn_b32 D:v1 A:v1 S0:v1 S1:v1 O:16
        ds 02d 11 ds_wrxchg_rtn_b32 D:v1 A:v1 S0:v1 O:16
        ds 02e 11 ds_wrxchg2_rtn_b32 D:v2 A:v1 S0:v1 S1:v1 O:8x2
        ds 02f 11 ds_wrxchg2st64_rtn_b32 D:v2 A:v1 S0:v1 S1:v1 O:8x2
        ds 030 11 ds_cmpst_rtn_b32 D:v1 A:v1 S0:v1 S1:v1 O:16
        ds 031 11 ds_cmpst_rtn_f32 D:v1 A:v1 S0:v1 S1:v1 O:16
        ds 032 11 ds_min_rtn_f32 D:v1 A:v1 S0:v1 O:16
        ds 033 11 ds_max_rtn_f32 D:v1 A:v1 S0:v1 O:16
        ds 034 11 ds_wrap_rtn_b32 D:v1 A:v1 S0:v1 S1:v1 O:16
        ds 035 11 ds_swizzle_b32 D:v1 A:v1 O:swz
        ds 036 11 ds_read_b32 D:v1 A:v1 O:16
        ds 037 11 ds_read2_b32 D:v2 A:v1 O:8x2
        ds 038 11 ds_read2st64_b32 D:v2 A:v1 O:8x2
        ds 039 11 ds_read_i8 D:v1 A:v1 O:16
        ds 03a 11 ds_read_u8 D:v1 A:v1 O:16
        ds 03b 11 ds_read_i16 D:v1 A:v1 O:16
        ds 03c 11 ds_read_u16 D:v1 A:v1 O:16
        ds 03d 11 ds_consume D:v1 O:16
        ds 03e 11 ds_append D:v1 O:16
        ds 040 11 ds_add_u64 A:v1 S0:v2 O:16
        ds 041 11 ds_sub_u64 A:v1 S0:v2 O:16
        ds 042 11 ds_rsub_u64 A:v1 S0:v2 O:16
        ds 043 11 ds_inc_u64 A:v1 S0:v2 O:16
        ds 044 11 ds_dec_u64 A:v1 S0:v2 O:16
        ds 045 11 ds_min_i64 A:v1 S0:v2 O:16
        ds 046 11 ds_max_i64 A:v1 S0:v2 O:16
        ds 047 11 ds_min_u64 A:v1 S0:v2 O:16
        ds 048 11 ds_max_u64 A:v1 S0:v2 O:16
        ds 049 11 ds_and_b64 A:v1 S0:v2 O:16
        ds 04a 11 ds_or_b64 A:v1 S0:v2 O:16
        ds 04b 11 ds_xor_b64 A:v1 S0:v2 O:16
        ds 04c 11 ds_mskor_b64 A:v1 S0:v2 S1:v2 O:16
        ds 04d 11 ds_write_b64 A:v1 S0:v2 O:16
        ds 04e 11 ds_write2_b64 A:v1 S0:v2 S1:v2 O:8x2
        ds 04f 11 ds_write2st64_b64 A:v1 S0:v2 S1:v2 O:8x2
        ds 050 11 ds_cmpst_b64 A:v1 S0:v2 S1:v2 O:16
        ds 051 11 ds_cmpst_f64 A:v1 S0:v2 S1:v2 O:16
        ds 052 11 ds_min_f64 A:v1 S0:v2 O:16
        ds 053 11 ds_max_f64 A:v1 S0:v2 O:16
        ds 055 11 ds_add_rtn_f32 D:v1 A:v1 S0:v1 O:16
        ds 060 11 ds_add_rtn_u64 D:v2 A:v1 S0:v2 O:16
        ds 061 11 ds_sub_rtn_u64 D:v2 A:v1 S0:v2 O:16
        ds 062 11 ds_rsub_rtn_u64 D:v2 A:v1 S0:v2 O:16
        ds 063 11 ds_inc_rtn_u64 D:v2 A:v1 S0:v2 O:16
        ds 064 11 ds_dec_rtn_u64 D:v2 A:v1 S0:v2 O:16
        ds 065 11 ds_min_rtn_i64 D:v2 A:v1 S0:v2 O:16
        ds 066 11 ds_max_rtn_i64 D:v2 A:v1 S0:v2 O:16
        ds 067 11 ds_min_rtn_u64 D:v2 A:v1 S0:v2 O:16
        ds 068 11 ds_max_rtn_u64 D:v2 A:v1 S0:v2 O:16
        ds 069 11 ds_and_rtn_b64 D:v2 A:v1 S0:v2 O:16
        ds 06a 11 ds_or_rtn_b64 D:v2 A:v1 S0:v2 O:16
        ds 06b 11 ds_xor_rtn_b64 D:v2 A:v1 S0:v2 O:16
        ds 06c 11 ds_mskor_rtn_b64 D:v2 A:v1 S0:v2 S1:v2 O:16
        ds 06d 11 ds_wrxchg_rtn_b64 D:v2 A:v1 S0:v2 O:16
        ds 06e 11 ds_wrxchg2_rtn_b64 D:v4 A:v1 S0:v2 S1:v2 O:8x2
        ds 06f 11 ds_wrxchg2st64_rtn_b64 D:v4 A:v1 S0:v2 S1:v2 O:8x2
        ds 070 11 ds_cmpst_rtn_b64 D:v2 A:v1 S0:v2 S1:v2 O:16
        ds 071 11 ds_cmpst_rtn_f64 D:v2 A:v1 S0:v2 S1:v2 O:16
        ds 072 11 ds_min_rtn_f64 D:v2 A:v1 S0:v2 O:16
        ds 073 11 ds_max_rtn_f64 D:v2 A:v1 S0:v2 O:16
        ds 076 11 ds_read_b64 D:v2 A:v1 O:16
        ds 077 11 ds_read2_b64 D:v4 A:v1 O:8x2
        ds 078 11 ds_read2st64_b64 D:v4 A:v1 O:8x2
        ds 07e 11 ds_condxchg32_rtn_b64 D:v2 A:v1 S0:v2 O:16
        ds 080 10 ds_add_src2_u32 A:v1 O:16
        ds 081 10 ds_sub_src2_u32 A:v1 O:16
        ds 082 10 ds_rsub_src2_u32 A:v1 O:16
        ds 083 10 ds_inc_src2_u32 A:v1 O:16
        ds 084 10 ds_dec_src2_u32 A:v1 O:16
        ds 085 10 ds_min_src2_i32 A:v1 O:16
        ds 086 10 ds_max_src2_i32 A:v1 O:16
        ds 087 10 ds_min_src2_u32 A:v1 O:16
        ds 088 10 ds_max_src2_u32 A:v1 O:16
        ds 089 10 ds_and_src2_b32 A:v1 O:16
        ds 08a 10 ds_or_src2_b32 A:v1 O:16
        ds 08b 10 ds_xor_src2_b32 A:v1 O:16
        ds 08d 10 ds_write_src2_b32 A:v1 O:16
        ds 092 10 ds_min_src2_f32 A:v1 O:16
        ds 093 10 ds_max_src2_f32 A:v1 O:16
        ds 095 10 ds_add_src2_f32 A:v1 O:16
        ds 0a0 11 ds_write_b8_d16_hi A:v1 S0:v1 O:16
        ds 0a1 11 ds_write_b16_d16_hi A:v1 S0:v1 O:16
        ds 0a2 11 ds_read_u8_d16 D:v1 A:v1 O:16
        ds 0a3 11 ds_read_u8_d16_hi D:v1 A:v1 O:16
        ds 0a4 11 ds_read_i8_d16 D:v1 A:v1 O:16
        ds 0a5 11 ds_read_i8_d16_hi D:v1 A:v1 O:16
        ds 0a6 11 ds_read_u16_d16 D:v1 A:v1 O:16
        ds 0a7 11 ds_read_u16_d16_hi D:v1 A:v1 O:16
        ds 0b0 11 ds_write_addtid_b32 S0:v1 O:16
        ds 0b1 11 ds_read_addtid_b32 D:v1 O:16
        ds 0b2 11 ds_permute_b32 D:v1 A:v1 S0:v1 O:16
        ds 0b3 11 ds_bpermute_b32 D:v1 A:v1 S0:v1 O:16
        ds 0c0 10 ds_add_src2_u64 A:v1 O:16
        ds 0c1 10 ds_sub_src2_u64 A:v1 O:16
        ds 0c2 10 ds_rsub_src2_u64 A:v1 O:16
        ds 0c3 10 ds_inc_src2_u64 A:v1 O:16
        ds 0c4 10 ds_dec_src2_u64 A:v1 O:16
        ds 0c5 10 ds_min_src2_i64 A:v1 O:16
        ds 0c6 10 ds_max_src2_i64 A:v1 O:16
        ds 0c7 10 ds_min_src2_u64 A:v1 O:16
        ds 0c8 10 ds_max_src2_u64 A:v1 O:16
        ds 0c9 10 ds_and_src2_b64 A:v1 O:16
        ds 0ca 10 ds_or_src2_b64 A:v1 O:16
        ds 0cb 10 ds_xor_src2_b64 A:v1 O:16
        ds 0cd 10 ds_write_src2_b64 A:v1 O:16
        ds 0d2 10 ds_min_src2_f64 A:v1 O:16
        ds 0d3 10 ds_max_src2_f64 A:v1 O:16
        ds 0de 11 ds_write_b96 A:v1 S0:v3 O:16
        ds 0df 11 ds_write_b128 A:v1 S0:v4 O:16
        ds 0fe 11 ds_read_b96 D:v3 A:v1 O:16
        ds 0ff 11 ds_read_b128 D:v4 A:v1 O:16
        flat 008 11 flat_load_ubyte D:v1 A:v2 O:16
        flat 009 11 flat_load_sbyte D:v1 A:v2 O:16
        flat 00a 11 flat_load_ushort D:v1 A:v2 O:16
        flat 00b 11 flat_load_sshort D:v1 A:v2 O:16
        flat 00c 11 flat_load_dword D:v1 A:v2 O:16
        flat 00d 11 flat_load_dwordx2 D:v2 A:v2 O:16
        flat 00e 11 flat_load_dwordx4 D:v4 A:v2 O:16
        flat 00f 11 flat_load_dwordx3 D:v3 A:v2 O:16
        flat 018 11 flat_store_byte A:v2 S0:v1 O:16
        flat 019 11 flat_store_byte_d16_hi A:v2 S0:v1 O:16
        flat 01a 11 flat_store_short A:v2 S0:v1 O:16
        flat 01b 11 flat_store_short_d16_hi A:v2 S0:v1 O:16
        flat 01c 11 flat_store_dword A:v2 S0:v1 O:16
        flat 01d 11 flat_store_dwordx2 A:v2 S0:v2 O:16
        flat 01e 11 flat_store_dwordx4 A:v2 S0:v4 O:16
        flat 01f 11 flat_store_dwordx3 A:v2 S0:v3 O:16
        flat 020 11 flat_load_ubyte_d16 D:v1 A:v2 O:16
        flat 021 11 flat_load_ubyte_d16_hi D:v1 A:v2 O:16
        flat 022 11 flat_load_sbyte_d16 D:v1 A:v2 O:16
        flat 023 11 flat_load_sbyte_d16_hi D:v1 A:v2 O:16
        flat 024 11 flat_load_short_d16 D:v1 A:v2 O:16
        flat 025 11 flat_load_short_d16_hi D:v1 A:v2 O:16
        flat 030 11 flat_atomic_swap A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 031 11 flat_atomic_cmpswap A:v2 S0:v2 O:16 | D:v1 A:v2 S0:v2 O:16
        flat 032 11 flat_atomic_add A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 033 11 flat_atomic_sub A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 035 11 flat_atomic_smin A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 036 11 flat_atomic_umin A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 037 11 flat_atomic_smax A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 038 11 flat_atomic_umax A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 039 11 flat_atomic_and A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 03a 11 flat_atomic_or A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 03b 11 flat_atomic_xor A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 03c 11 flat_atomic_inc A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 03d 11 flat_atomic_dec A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 03e 11 flat_atomic_fcmpswap A:v2 S0:v2 O:16 | D:v1 A:v2 S0:v2 O:16
        flat 03f 11 flat_atomic_fmin A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 040 11 flat_atomic_fmax A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 050 11 flat_atomic_swap_x2 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 051 11 flat_atomic_cmpswap_x2 A:v2 S0:v4 O:16 | D:v2 A:v2 S0:v4 O:16
        flat 052 11 flat_atomic_add_x2 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 053 11 flat_atomic_sub_x2 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 055 11 flat_atomic_smin_x2 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 056 11 flat_atomic_umin_x2 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 057 11 flat_atomic_smax_x2 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 058 11 flat_atomic_umax_x2 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 059 11 flat_atomic_and_x2 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 05a 11 flat_atomic_or_x2 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 05b 11 flat_atomic_xor_x2 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 05c 11 flat_atomic_inc_x2 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 05d 11 flat_atomic_dec_x2 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 05e 11 flat_atomic_fcmpswap_x2 A:v2 S0:v4 O:16 | D:v2 A:v2 S0:v4 O:16
        flat 05f 11 flat_atomic_fmin_x2 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 060 11 flat_atomic_fmax_x2 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        global 008 11 global_load_ubyte D:v1 A:v2 =off O:16
        global 009 11 global_load_sbyte D:v1 A:v2 =off O:16
        global 00a 11 global_load_ushort D:v1 A:v2 =off O:16
        global 00b 11 global_load_sshort D:v1 A:v2 =off O:16
        global 00c 11 global_load_dword D:v1 A:v2 =off O:16
        global 00d 11 global_load_dwordx2 D:v2 A:v2 =off O:16
        global 00e 11 global_load_dwordx4 D:v4 A:v2 =off O:16
        global 00f 11 global_load_dwordx3 D:v3 A:v2 =off O:16
        global 016 01 global_load_dword_addtid D:v1 =off O:16
        global 017 01 global_store_dword_addtid S0:v1 =off O:16
        global 018 11 global_store_byte A:v2 S0:v1 =off O:16
        global 019 11 global_store_byte_d16_hi A:v2 S0:v1 =off O:16
        global 01a 11 global_store_short A:v2 S0:v1 =off O:16
        global 01b 11 global_store_short_d16_hi A:v2 S0:v1 =off O:16
        global 01c 11 global_store_dword A:v2 S0:v1 =off O:16
        global 01d 11 global_store_dwordx2 A:v2 S0:v2 =off O:16
        global 01e 11 global_store_dwordx4 A:v2 S0:v4 =off O:16
        global 01f 11 global_store_dwordx3 A:v2 S0:v3 =off O:16
        global 020 11 global_load_ubyte_d16 D:v1 A:v2 =off O:16
        global 021 11 global_load_ubyte_d16_hi D:v1 A:v2 =off O:16
        global 022 11 global_load_sbyte_d16 D:v1 A:v2 =off O:16
        global 023 11 global_load_sbyte_d16_hi D:v1 A:v2 =off O:16
        global 024 11 global_load_short_d16 D:v1 A:v2 =off O:16
        global 025 11 global_load_short_d16_hi D:v1 A:v2 =off O:16
        global 030 11 global_atomic_swap A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 031 11 global_atomic_cmpswap A:v2 S0:v2 =off O:16 | D:v1 A:v2 S0:v2 =off O:16
        global 032 11 global_atomic_add A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 033 11 global_atomic_sub A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 034 01 global_atomic_csub A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 035 11 global_atomic_smin A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 036 11 global_atomic_umin A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 037 11 global_atomic_smax A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 038 11 global_atomic_umax A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 039 11 global_atomic_and A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 03a 11 global_atomic_or A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 03b 11 global_atomic_xor A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 03c 11 global_atomic_inc A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 03d 11 global_atomic_dec A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 03e 11 global_atomic_fcmpswap A:v2 S0:v2 =off O:16 | D:v1 A:v2 S0:v2 =off O:16
        global 03f 11 global_atomic_fmin A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 040 11 global_atomic_fmax A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 050 11 global_atomic_swap_x2 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 051 11 global_atomic_cmpswap_x2 A:v2 S0:v4 =off O:16 | D:v2 A:v2 S0:v4 =off O:16
        global 052 11 global_atomic_add_x2 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 053 11 global_atomic_sub_x2 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 055 11 global_atomic_smin_x2 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 056 11 global_atomic_umin_x2 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 057 11 global_atomic_smax_x2 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 058 11 global_atomic_umax_x2 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 059 11 global_atomic_and_x2 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 05a 11 global_atomic_or_x2 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 05b 11 global_atomic_xor_x2 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 05c 11 global_atomic_inc_x2 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 05d 11 global_atomic_dec_x2 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 05e 11 global_atomic_fcmpswap_x2 A:v2 S0:v4 =off O:16 | D:v2 A:v2 S0:v4 =off O:16
        global 05f 11 global_atomic_fmin_x2 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 060 11 global_atomic_fmax_x2 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        mubuf 071 11 buffer_gl0_inv -
        mubuf 072 11 buffer_gl1_inv -
        """;
}
