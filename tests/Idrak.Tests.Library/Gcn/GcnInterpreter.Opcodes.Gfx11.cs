// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Tests.Gcn;

// Made by tests/Idrak.Tests/data/gcn/interpreter/make_opcode_table.py from LLVM 18.1.8's disassembler; do not edit, rerun the
// script. One opcode a line: the encoding, the opcode (hex), the CPUs that have it (gfx1100, gfx1102, 1 or 0 each), the mnemonic
// as LLVM prints it, and the operand template in print order (a field, its register class and width; '=' a fixed
// operand; K an immediate and its kind; O the offset's kind); after '|', FLAT's template when glc is set.
// The gfx11 family in wave64, in its own word layouts (plan 35, G8a).
internal static partial class GcnOpcodeTable
{
    internal const string Gfx11Cpus = "gfx1100 gfx1102";

    internal const string Gfx11Text = """
        sop1 000 11 s_mov_b32 D:s1 S0:s1
        sop1 001 11 s_mov_b64 D:s2 S0:s2
        sop1 002 11 s_cmov_b32 D:s1 S0:s1
        sop1 003 11 s_cmov_b64 D:s2 S0:s2
        sop1 004 11 s_brev_b32 D:s1 S0:s1
        sop1 005 11 s_brev_b64 D:s2 S0:s2
        sop1 008 11 s_ctz_i32_b32 D:s1 S0:s1
        sop1 009 11 s_ctz_i32_b64 D:s1 S0:s2
        sop1 00a 11 s_clz_i32_u32 D:s1 S0:s1
        sop1 00b 11 s_clz_i32_u64 D:s1 S0:s2
        sop1 00c 11 s_cls_i32 D:s1 S0:s1
        sop1 00d 11 s_cls_i32_i64 D:s1 S0:s2
        sop1 00e 11 s_sext_i32_i8 D:s1 S0:s1
        sop1 00f 11 s_sext_i32_i16 D:s1 S0:s1
        sop1 010 11 s_bitset0_b32 D:s1 S0:s1
        sop1 011 11 s_bitset0_b64 D:s2 S0:s1
        sop1 012 11 s_bitset1_b32 D:s1 S0:s1
        sop1 013 11 s_bitset1_b64 D:s2 S0:s1
        sop1 014 11 s_bitreplicate_b64_b32 D:s2 S0:s1
        sop1 015 11 s_abs_i32 D:s1 S0:s1
        sop1 016 11 s_bcnt0_i32_b32 D:s1 S0:s1
        sop1 017 11 s_bcnt0_i32_b64 D:s1 S0:s2
        sop1 018 11 s_bcnt1_i32_b32 D:s1 S0:s1
        sop1 019 11 s_bcnt1_i32_b64 D:s1 S0:s2
        sop1 01a 11 s_quadmask_b32 D:s1 S0:s1
        sop1 01b 11 s_quadmask_b64 D:s2 S0:s2
        sop1 01c 11 s_wqm_b32 D:s1 S0:s1
        sop1 01d 11 s_wqm_b64 D:s2 S0:s2
        sop1 01e 11 s_not_b32 D:s1 S0:s1
        sop1 01f 11 s_not_b64 D:s2 S0:s2
        sop1 020 11 s_and_saveexec_b32 D:s1 S0:s1
        sop1 021 11 s_and_saveexec_b64 D:s2 S0:s2
        sop1 022 11 s_or_saveexec_b32 D:s1 S0:s1
        sop1 023 11 s_or_saveexec_b64 D:s2 S0:s2
        sop1 024 11 s_xor_saveexec_b32 D:s1 S0:s1
        sop1 025 11 s_xor_saveexec_b64 D:s2 S0:s2
        sop1 026 11 s_nand_saveexec_b32 D:s1 S0:s1
        sop1 027 11 s_nand_saveexec_b64 D:s2 S0:s2
        sop1 028 11 s_nor_saveexec_b32 D:s1 S0:s1
        sop1 029 11 s_nor_saveexec_b64 D:s2 S0:s2
        sop1 02a 11 s_xnor_saveexec_b32 D:s1 S0:s1
        sop1 02b 11 s_xnor_saveexec_b64 D:s2 S0:s2
        sop1 02c 11 s_and_not0_saveexec_b32 D:s1 S0:s1
        sop1 02d 11 s_and_not0_saveexec_b64 D:s2 S0:s2
        sop1 02e 11 s_or_not0_saveexec_b32 D:s1 S0:s1
        sop1 02f 11 s_or_not0_saveexec_b64 D:s2 S0:s2
        sop1 030 11 s_and_not1_saveexec_b32 D:s1 S0:s1
        sop1 031 11 s_and_not1_saveexec_b64 D:s2 S0:s2
        sop1 032 11 s_or_not1_saveexec_b32 D:s1 S0:s1
        sop1 033 11 s_or_not1_saveexec_b64 D:s2 S0:s2
        sop1 034 11 s_and_not0_wrexec_b32 D:s1 S0:s1
        sop1 035 11 s_and_not0_wrexec_b64 D:s2 S0:s2
        sop1 036 11 s_and_not1_wrexec_b32 D:s1 S0:s1
        sop1 037 11 s_and_not1_wrexec_b64 D:s2 S0:s2
        sop1 040 11 s_movrels_b32 D:s1 S0:s1
        sop1 041 11 s_movrels_b64 D:s2 S0:s2
        sop1 042 11 s_movreld_b32 D:s1 S0:s1
        sop1 043 11 s_movreld_b64 D:s2 S0:s2
        sop1 044 11 s_movrelsd_2_b32 D:s1 S0:s1
        sop1 047 11 s_getpc_b64 D:s2
        sop1 048 11 s_setpc_b64 S0:s2
        sop1 049 11 s_swappc_b64 D:s2 S0:s2
        sop1 04a 11 s_rfe_b64 S0:s2
        sop1 04c 11 s_sendmsg_rtn_b32 D:s1 K:sendmsg
        sop1 04d 11 s_sendmsg_rtn_b64 D:s2 K:sendmsg
        sop2 000 11 s_add_u32 D:s1 S0:s1 S1:s1
        sop2 001 11 s_sub_u32 D:s1 S0:s1 S1:s1
        sop2 002 11 s_add_i32 D:s1 S0:s1 S1:s1
        sop2 003 11 s_sub_i32 D:s1 S0:s1 S1:s1
        sop2 004 11 s_addc_u32 D:s1 S0:s1 S1:s1
        sop2 005 11 s_subb_u32 D:s1 S0:s1 S1:s1
        sop2 006 11 s_absdiff_i32 D:s1 S0:s1 S1:s1
        sop2 008 11 s_lshl_b32 D:s1 S0:s1 S1:s1
        sop2 009 11 s_lshl_b64 D:s2 S0:s2 S1:s1
        sop2 00a 11 s_lshr_b32 D:s1 S0:s1 S1:s1
        sop2 00b 11 s_lshr_b64 D:s2 S0:s2 S1:s1
        sop2 00c 11 s_ashr_i32 D:s1 S0:s1 S1:s1
        sop2 00d 11 s_ashr_i64 D:s2 S0:s2 S1:s1
        sop2 00e 11 s_lshl1_add_u32 D:s1 S0:s1 S1:s1
        sop2 00f 11 s_lshl2_add_u32 D:s1 S0:s1 S1:s1
        sop2 010 11 s_lshl3_add_u32 D:s1 S0:s1 S1:s1
        sop2 011 11 s_lshl4_add_u32 D:s1 S0:s1 S1:s1
        sop2 012 11 s_min_i32 D:s1 S0:s1 S1:s1
        sop2 013 11 s_min_u32 D:s1 S0:s1 S1:s1
        sop2 014 11 s_max_i32 D:s1 S0:s1 S1:s1
        sop2 015 11 s_max_u32 D:s1 S0:s1 S1:s1
        sop2 016 11 s_and_b32 D:s1 S0:s1 S1:s1
        sop2 017 11 s_and_b64 D:s2 S0:s2 S1:s2
        sop2 018 11 s_or_b32 D:s1 S0:s1 S1:s1
        sop2 019 11 s_or_b64 D:s2 S0:s2 S1:s2
        sop2 01a 11 s_xor_b32 D:s1 S0:s1 S1:s1
        sop2 01b 11 s_xor_b64 D:s2 S0:s2 S1:s2
        sop2 01c 11 s_nand_b32 D:s1 S0:s1 S1:s1
        sop2 01d 11 s_nand_b64 D:s2 S0:s2 S1:s2
        sop2 01e 11 s_nor_b32 D:s1 S0:s1 S1:s1
        sop2 01f 11 s_nor_b64 D:s2 S0:s2 S1:s2
        sop2 020 11 s_xnor_b32 D:s1 S0:s1 S1:s1
        sop2 021 11 s_xnor_b64 D:s2 S0:s2 S1:s2
        sop2 022 11 s_and_not1_b32 D:s1 S0:s1 S1:s1
        sop2 023 11 s_and_not1_b64 D:s2 S0:s2 S1:s2
        sop2 024 11 s_or_not1_b32 D:s1 S0:s1 S1:s1
        sop2 025 11 s_or_not1_b64 D:s2 S0:s2 S1:s2
        sop2 026 11 s_bfe_u32 D:s1 S0:s1 S1:s1
        sop2 027 11 s_bfe_i32 D:s1 S0:s1 S1:s1
        sop2 028 11 s_bfe_u64 D:s2 S0:s2 S1:s1
        sop2 029 11 s_bfe_i64 D:s2 S0:s2 S1:s1
        sop2 02a 11 s_bfm_b32 D:s1 S0:s1 S1:s1
        sop2 02b 11 s_bfm_b64 D:s2 S0:s1 S1:s1
        sop2 02c 11 s_mul_i32 D:s1 S0:s1 S1:s1
        sop2 02d 11 s_mul_hi_u32 D:s1 S0:s1 S1:s1
        sop2 02e 11 s_mul_hi_i32 D:s1 S0:s1 S1:s1
        sop2 030 11 s_cselect_b32 D:s1 S0:s1 S1:s1
        sop2 031 11 s_cselect_b64 D:s2 S0:s2 S1:s2
        sop2 032 11 s_pack_ll_b32_b16 D:s1 S0:s1 S1:s1
        sop2 033 11 s_pack_lh_b32_b16 D:s1 S0:s1 S1:s1
        sop2 034 11 s_pack_hh_b32_b16 D:s1 S0:s1 S1:s1
        sop2 035 11 s_pack_hl_b32_b16 D:s1 S0:s1 S1:s1
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
        sopk 011 11 s_getreg_b32 D:s1 K:hwreg
        sopk 012 11 s_setreg_b32 K:hwreg D:s1
        sopk 013 11 s_setreg_imm32_b32 K:hwreg K:lit
        sopk 014 11 s_call_b64 D:s2 K:dec
        sopk 016 11 s_subvector_loop_begin D:s1 K:dec
        sopk 017 11 s_subvector_loop_end D:s1 K:dec
        sopk 018 11 s_waitcnt_vscnt D:s1 K:hex16
        sopk 019 11 s_waitcnt_vmcnt D:s1 K:hex16
        sopk 01a 11 s_waitcnt_expcnt D:s1 K:hex16
        sopk 01b 11 s_waitcnt_lgkmcnt D:s1 K:hex16
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
        sopc 010 11 s_cmp_eq_u64 S0:s2 S1:s2
        sopc 011 11 s_cmp_lg_u64 S0:s2 S1:s2
        sopp 000 11 s_nop K:dec
        sopp 001 11 s_setkill K:dec
        sopp 002 11 s_sethalt K:dec
        sopp 003 11 s_sleep K:dec
        sopp 004 11 s_set_inst_prefetch_distance K:hex16
        sopp 005 11 s_clause K:hex16
        sopp 007 11 s_delay_alu K:delay
        sopp 008 11 s_waitcnt_depctr K:hex16
        sopp 009 11 s_waitcnt K:waitcnt
        sopp 00a 11 s_wait_idle -
        sopp 00b 11 s_wait_event K:hex16
        sopp 010 11 s_trap K:dec
        sopp 011 11 s_round_mode K:hex16
        sopp 012 11 s_denorm_mode K:dec
        sopp 01f 11 s_code_end -
        sopp 020 11 s_branch K:dec
        sopp 021 11 s_cbranch_scc0 K:dec
        sopp 022 11 s_cbranch_scc1 K:dec
        sopp 023 11 s_cbranch_vccz K:dec
        sopp 024 11 s_cbranch_vccnz K:dec
        sopp 025 11 s_cbranch_execz K:dec
        sopp 026 11 s_cbranch_execnz K:dec
        sopp 027 11 s_cbranch_cdbgsys K:dec
        sopp 028 11 s_cbranch_cdbguser K:dec
        sopp 029 11 s_cbranch_cdbgsys_or_user K:dec
        sopp 02a 11 s_cbranch_cdbgsys_and_user K:dec
        sopp 030 11 s_endpgm K:dec
        sopp 031 11 s_endpgm_saved -
        sopp 034 11 s_wakeup -
        sopp 035 11 s_setprio K:dec
        sopp 036 11 s_sendmsg K:sendmsg
        sopp 037 11 s_sendmsghalt K:sendmsg
        sopp 038 11 s_incperflevel K:dec
        sopp 039 11 s_decperflevel K:dec
        sopp 03a 11 s_ttracedata -
        sopp 03b 11 s_ttracedata_imm K:hex16
        sopp 03c 11 s_icache_inv -
        sopp 03d 11 s_barrier -
        smem 000 11 s_load_b32 D:s1 B:s2 O:hex
        smem 001 11 s_load_b64 D:s2 B:s2 O:hex
        smem 002 11 s_load_b128 D:s4 B:s2 O:hex
        smem 003 11 s_load_b256 D:s8 B:s2 O:hex
        smem 004 11 s_load_b512 D:s16 B:s2 O:hex
        smem 008 11 s_buffer_load_b32 D:s1 B:s4 O:hex
        smem 009 11 s_buffer_load_b64 D:s2 B:s4 O:hex
        smem 00a 11 s_buffer_load_b128 D:s4 B:s4 O:hex
        smem 00b 11 s_buffer_load_b256 D:s8 B:s4 O:hex
        smem 00c 11 s_buffer_load_b512 D:s16 B:s4 O:hex
        smem 020 11 s_gl1_inv -
        smem 021 11 s_dcache_inv -
        smem 022 11 s_atc_probe K:dec B:s2 O:hex
        smem 023 11 s_atc_probe_buffer K:dec B:s4 O:hex
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
        vop1 00c 11 v_cvt_nearest_i32_f32_e32 D:v1 S0:v1
        vop1 00d 11 v_cvt_floor_i32_f32_e32 D:v1 S0:v1
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
        vop1 039 11 v_clz_i32_u32_e32 D:v1 S0:v1
        vop1 03a 11 v_ctz_i32_b32_e32 D:v1 S0:v1
        vop1 03b 11 v_cls_i32_e32 D:v1 S0:v1
        vop1 03c 11 v_frexp_exp_i32_f64_e32 D:v1 S0:v2
        vop1 03d 11 v_frexp_mant_f64_e32 D:v2 S0:v2
        vop1 03e 11 v_fract_f64_e32 D:v2 S0:v2
        vop1 03f 11 v_frexp_exp_i32_f32_e32 D:v1 S0:v1
        vop1 040 11 v_frexp_mant_f32_e32 D:v1 S0:v1
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
        vop1 067 11 v_permlane64_b32 D:v1 S0:v1
        vop1 068 11 v_swaprel_b32 D:v1 S0:v1
        vop1 069 11 v_not_b16_e32 D:v1 S0:v1
        vop1 06a 11 v_cvt_i32_i16_e32 D:v1 S0:v1
        vop1 06b 11 v_cvt_u32_u16_e32 D:v1 S0:v1
        vop2 001 11 v_cndmask_b32_e32 D:v1 S0:v1 S1:v1 =vcc
        vop2 002 11 v_dot2acc_f32_f16 D:v1 S0:v1 S1:v1
        vop2 003 11 v_add_f32_e32 D:v1 S0:v1 S1:v1
        vop2 004 11 v_sub_f32_e32 D:v1 S0:v1 S1:v1
        vop2 005 11 v_subrev_f32_e32 D:v1 S0:v1 S1:v1
        vop2 006 11 v_fmac_dx9_zero_f32_e32 D:v1 S0:v1 S1:v1
        vop2 007 11 v_mul_dx9_zero_f32_e32 D:v1 S0:v1 S1:v1
        vop2 008 11 v_mul_f32_e32 D:v1 S0:v1 S1:v1
        vop2 009 11 v_mul_i32_i24_e32 D:v1 S0:v1 S1:v1
        vop2 00a 11 v_mul_hi_i32_i24_e32 D:v1 S0:v1 S1:v1
        vop2 00b 11 v_mul_u32_u24_e32 D:v1 S0:v1 S1:v1
        vop2 00c 11 v_mul_hi_u32_u24_e32 D:v1 S0:v1 S1:v1
        vop2 00f 11 v_min_f32_e32 D:v1 S0:v1 S1:v1
        vop2 010 11 v_max_f32_e32 D:v1 S0:v1 S1:v1
        vop2 011 11 v_min_i32_e32 D:v1 S0:v1 S1:v1
        vop2 012 11 v_max_i32_e32 D:v1 S0:v1 S1:v1
        vop2 013 11 v_min_u32_e32 D:v1 S0:v1 S1:v1
        vop2 014 11 v_max_u32_e32 D:v1 S0:v1 S1:v1
        vop2 018 11 v_lshlrev_b32_e32 D:v1 S0:v1 S1:v1
        vop2 019 11 v_lshrrev_b32_e32 D:v1 S0:v1 S1:v1
        vop2 01a 11 v_ashrrev_i32_e32 D:v1 S0:v1 S1:v1
        vop2 01b 11 v_and_b32_e32 D:v1 S0:v1 S1:v1
        vop2 01c 11 v_or_b32_e32 D:v1 S0:v1 S1:v1
        vop2 01d 11 v_xor_b32_e32 D:v1 S0:v1 S1:v1
        vop2 01e 11 v_xnor_b32_e32 D:v1 S0:v1 S1:v1
        vop2 020 11 v_add_co_ci_u32_e32 D:v1 =vcc S0:v1 S1:v1 =vcc
        vop2 021 11 v_sub_co_ci_u32_e32 D:v1 =vcc S0:v1 S1:v1 =vcc
        vop2 022 11 v_subrev_co_ci_u32_e32 D:v1 =vcc S0:v1 S1:v1 =vcc
        vop2 025 11 v_add_nc_u32_e32 D:v1 S0:v1 S1:v1
        vop2 026 11 v_sub_nc_u32_e32 D:v1 S0:v1 S1:v1
        vop2 027 11 v_subrev_nc_u32_e32 D:v1 S0:v1 S1:v1
        vop2 02b 11 v_fmac_f32_e32 D:v1 S0:v1 S1:v1
        vop2 02c 11 v_fmamk_f32 D:v1 S0:v1 K:lit S1:v1
        vop2 02d 11 v_fmaak_f32 D:v1 S0:v1 S1:v1 K:lit
        vop2 02f 11 v_cvt_pk_rtz_f16_f32_e32 D:v1 S0:v1 S1:v1
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
        vop2 03e 11 v_cmp_nlg_f16_e32 =vcc S0:v1 S1:v1
        vopc 000 11 v_cmp_f_f16_e32 =vcc S0:v1 S1:v1
        vopc 001 11 v_cmp_lt_f16_e32 =vcc S0:v1 S1:v1
        vopc 002 11 v_cmp_eq_f16_e32 =vcc S0:v1 S1:v1
        vopc 003 11 v_cmp_le_f16_e32 =vcc S0:v1 S1:v1
        vopc 004 11 v_cmp_gt_f16_e32 =vcc S0:v1 S1:v1
        vopc 005 11 v_cmp_lg_f16_e32 =vcc S0:v1 S1:v1
        vopc 006 11 v_cmp_ge_f16_e32 =vcc S0:v1 S1:v1
        vopc 007 11 v_cmp_o_f16_e32 =vcc S0:v1 S1:v1
        vopc 008 11 v_cmp_u_f16_e32 =vcc S0:v1 S1:v1
        vopc 009 11 v_cmp_nge_f16_e32 =vcc S0:v1 S1:v1
        vopc 00a 11 v_cmp_nlg_f16_e32 =vcc S0:v1 S1:v1
        vopc 00b 11 v_cmp_ngt_f16_e32 =vcc S0:v1 S1:v1
        vopc 00c 11 v_cmp_nle_f16_e32 =vcc S0:v1 S1:v1
        vopc 00d 11 v_cmp_neq_f16_e32 =vcc S0:v1 S1:v1
        vopc 00e 11 v_cmp_nlt_f16_e32 =vcc S0:v1 S1:v1
        vopc 00f 11 v_cmp_t_f16_e32 =vcc S0:v1 S1:v1
        vopc 010 11 v_cmp_f_f32_e32 =vcc S0:v1 S1:v1
        vopc 011 11 v_cmp_lt_f32_e32 =vcc S0:v1 S1:v1
        vopc 012 11 v_cmp_eq_f32_e32 =vcc S0:v1 S1:v1
        vopc 013 11 v_cmp_le_f32_e32 =vcc S0:v1 S1:v1
        vopc 014 11 v_cmp_gt_f32_e32 =vcc S0:v1 S1:v1
        vopc 015 11 v_cmp_lg_f32_e32 =vcc S0:v1 S1:v1
        vopc 016 11 v_cmp_ge_f32_e32 =vcc S0:v1 S1:v1
        vopc 017 11 v_cmp_o_f32_e32 =vcc S0:v1 S1:v1
        vopc 018 11 v_cmp_u_f32_e32 =vcc S0:v1 S1:v1
        vopc 019 11 v_cmp_nge_f32_e32 =vcc S0:v1 S1:v1
        vopc 01a 11 v_cmp_nlg_f32_e32 =vcc S0:v1 S1:v1
        vopc 01b 11 v_cmp_ngt_f32_e32 =vcc S0:v1 S1:v1
        vopc 01c 11 v_cmp_nle_f32_e32 =vcc S0:v1 S1:v1
        vopc 01d 11 v_cmp_neq_f32_e32 =vcc S0:v1 S1:v1
        vopc 01e 11 v_cmp_nlt_f32_e32 =vcc S0:v1 S1:v1
        vopc 01f 11 v_cmp_t_f32_e32 =vcc S0:v1 S1:v1
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
        vopc 02f 11 v_cmp_t_f64_e32 =vcc S0:v2 S1:v2
        vopc 031 11 v_cmp_lt_i16_e32 =vcc S0:v1 S1:v1
        vopc 032 11 v_cmp_eq_i16_e32 =vcc S0:v1 S1:v1
        vopc 033 11 v_cmp_le_i16_e32 =vcc S0:v1 S1:v1
        vopc 034 11 v_cmp_gt_i16_e32 =vcc S0:v1 S1:v1
        vopc 035 11 v_cmp_ne_i16_e32 =vcc S0:v1 S1:v1
        vopc 036 11 v_cmp_ge_i16_e32 =vcc S0:v1 S1:v1
        vopc 039 11 v_cmp_lt_u16_e32 =vcc S0:v1 S1:v1
        vopc 03a 11 v_cmp_eq_u16_e32 =vcc S0:v1 S1:v1
        vopc 03b 11 v_cmp_le_u16_e32 =vcc S0:v1 S1:v1
        vopc 03c 11 v_cmp_gt_u16_e32 =vcc S0:v1 S1:v1
        vopc 03d 11 v_cmp_ne_u16_e32 =vcc S0:v1 S1:v1
        vopc 03e 11 v_cmp_ge_u16_e32 =vcc S0:v1 S1:v1
        vopc 040 11 v_cmp_f_i32_e32 =vcc S0:v1 S1:v1
        vopc 041 11 v_cmp_lt_i32_e32 =vcc S0:v1 S1:v1
        vopc 042 11 v_cmp_eq_i32_e32 =vcc S0:v1 S1:v1
        vopc 043 11 v_cmp_le_i32_e32 =vcc S0:v1 S1:v1
        vopc 044 11 v_cmp_gt_i32_e32 =vcc S0:v1 S1:v1
        vopc 045 11 v_cmp_ne_i32_e32 =vcc S0:v1 S1:v1
        vopc 046 11 v_cmp_ge_i32_e32 =vcc S0:v1 S1:v1
        vopc 047 11 v_cmp_t_i32_e32 =vcc S0:v1 S1:v1
        vopc 048 11 v_cmp_f_u32_e32 =vcc S0:v1 S1:v1
        vopc 049 11 v_cmp_lt_u32_e32 =vcc S0:v1 S1:v1
        vopc 04a 11 v_cmp_eq_u32_e32 =vcc S0:v1 S1:v1
        vopc 04b 11 v_cmp_le_u32_e32 =vcc S0:v1 S1:v1
        vopc 04c 11 v_cmp_gt_u32_e32 =vcc S0:v1 S1:v1
        vopc 04d 11 v_cmp_ne_u32_e32 =vcc S0:v1 S1:v1
        vopc 04e 11 v_cmp_ge_u32_e32 =vcc S0:v1 S1:v1
        vopc 04f 11 v_cmp_t_u32_e32 =vcc S0:v1 S1:v1
        vopc 050 11 v_cmp_f_i64_e32 =vcc S0:v2 S1:v2
        vopc 051 11 v_cmp_lt_i64_e32 =vcc S0:v2 S1:v2
        vopc 052 11 v_cmp_eq_i64_e32 =vcc S0:v2 S1:v2
        vopc 053 11 v_cmp_le_i64_e32 =vcc S0:v2 S1:v2
        vopc 054 11 v_cmp_gt_i64_e32 =vcc S0:v2 S1:v2
        vopc 055 11 v_cmp_ne_i64_e32 =vcc S0:v2 S1:v2
        vopc 056 11 v_cmp_ge_i64_e32 =vcc S0:v2 S1:v2
        vopc 057 11 v_cmp_t_i64_e32 =vcc S0:v2 S1:v2
        vopc 058 11 v_cmp_f_u64_e32 =vcc S0:v2 S1:v2
        vopc 059 11 v_cmp_lt_u64_e32 =vcc S0:v2 S1:v2
        vopc 05a 11 v_cmp_eq_u64_e32 =vcc S0:v2 S1:v2
        vopc 05b 11 v_cmp_le_u64_e32 =vcc S0:v2 S1:v2
        vopc 05c 11 v_cmp_gt_u64_e32 =vcc S0:v2 S1:v2
        vopc 05d 11 v_cmp_ne_u64_e32 =vcc S0:v2 S1:v2
        vopc 05e 11 v_cmp_ge_u64_e32 =vcc S0:v2 S1:v2
        vopc 05f 11 v_cmp_t_u64_e32 =vcc S0:v2 S1:v2
        vopc 07d 11 v_cmp_class_f16_e32 =vcc S0:v1 S1:v1
        vopc 07e 11 v_cmp_class_f32_e32 =vcc S0:v1 S1:v1
        vopc 07f 11 v_cmp_class_f64_e32 =vcc S0:v2 S1:v1
        vopc 080 11 v_cmpx_f_f16_e32 S0:v1 S1:v1
        vopc 081 11 v_cmpx_lt_f16_e32 S0:v1 S1:v1
        vopc 082 11 v_cmpx_eq_f16_e32 S0:v1 S1:v1
        vopc 083 11 v_cmpx_le_f16_e32 S0:v1 S1:v1
        vopc 084 11 v_cmpx_gt_f16_e32 S0:v1 S1:v1
        vopc 085 11 v_cmpx_lg_f16_e32 S0:v1 S1:v1
        vopc 086 11 v_cmpx_ge_f16_e32 S0:v1 S1:v1
        vopc 087 11 v_cmpx_o_f16_e32 S0:v1 S1:v1
        vopc 088 11 v_cmpx_u_f16_e32 S0:v1 S1:v1
        vopc 089 11 v_cmpx_nge_f16_e32 S0:v1 S1:v1
        vopc 08a 11 v_cmpx_nlg_f16_e32 S0:v1 S1:v1
        vopc 08b 11 v_cmpx_ngt_f16_e32 S0:v1 S1:v1
        vopc 08c 11 v_cmpx_nle_f16_e32 S0:v1 S1:v1
        vopc 08d 11 v_cmpx_neq_f16_e32 S0:v1 S1:v1
        vopc 08e 11 v_cmpx_nlt_f16_e32 S0:v1 S1:v1
        vopc 08f 11 v_cmpx_t_f16_e32 S0:v1 S1:v1
        vopc 090 11 v_cmpx_f_f32_e32 S0:v1 S1:v1
        vopc 091 11 v_cmpx_lt_f32_e32 S0:v1 S1:v1
        vopc 092 11 v_cmpx_eq_f32_e32 S0:v1 S1:v1
        vopc 093 11 v_cmpx_le_f32_e32 S0:v1 S1:v1
        vopc 094 11 v_cmpx_gt_f32_e32 S0:v1 S1:v1
        vopc 095 11 v_cmpx_lg_f32_e32 S0:v1 S1:v1
        vopc 096 11 v_cmpx_ge_f32_e32 S0:v1 S1:v1
        vopc 097 11 v_cmpx_o_f32_e32 S0:v1 S1:v1
        vopc 098 11 v_cmpx_u_f32_e32 S0:v1 S1:v1
        vopc 099 11 v_cmpx_nge_f32_e32 S0:v1 S1:v1
        vopc 09a 11 v_cmpx_nlg_f32_e32 S0:v1 S1:v1
        vopc 09b 11 v_cmpx_ngt_f32_e32 S0:v1 S1:v1
        vopc 09c 11 v_cmpx_nle_f32_e32 S0:v1 S1:v1
        vopc 09d 11 v_cmpx_neq_f32_e32 S0:v1 S1:v1
        vopc 09e 11 v_cmpx_nlt_f32_e32 S0:v1 S1:v1
        vopc 09f 11 v_cmpx_t_f32_e32 S0:v1 S1:v1
        vopc 0a0 11 v_cmpx_f_f64_e32 S0:v2 S1:v2
        vopc 0a1 11 v_cmpx_lt_f64_e32 S0:v2 S1:v2
        vopc 0a2 11 v_cmpx_eq_f64_e32 S0:v2 S1:v2
        vopc 0a3 11 v_cmpx_le_f64_e32 S0:v2 S1:v2
        vopc 0a4 11 v_cmpx_gt_f64_e32 S0:v2 S1:v2
        vopc 0a5 11 v_cmpx_lg_f64_e32 S0:v2 S1:v2
        vopc 0a6 11 v_cmpx_ge_f64_e32 S0:v2 S1:v2
        vopc 0a7 11 v_cmpx_o_f64_e32 S0:v2 S1:v2
        vopc 0a8 11 v_cmpx_u_f64_e32 S0:v2 S1:v2
        vopc 0a9 11 v_cmpx_nge_f64_e32 S0:v2 S1:v2
        vopc 0aa 11 v_cmpx_nlg_f64_e32 S0:v2 S1:v2
        vopc 0ab 11 v_cmpx_ngt_f64_e32 S0:v2 S1:v2
        vopc 0ac 11 v_cmpx_nle_f64_e32 S0:v2 S1:v2
        vopc 0ad 11 v_cmpx_neq_f64_e32 S0:v2 S1:v2
        vopc 0ae 11 v_cmpx_nlt_f64_e32 S0:v2 S1:v2
        vopc 0af 11 v_cmpx_t_f64_e32 S0:v2 S1:v2
        vopc 0b1 11 v_cmpx_lt_i16_e32 S0:v1 S1:v1
        vopc 0b2 11 v_cmpx_eq_i16_e32 S0:v1 S1:v1
        vopc 0b3 11 v_cmpx_le_i16_e32 S0:v1 S1:v1
        vopc 0b4 11 v_cmpx_gt_i16_e32 S0:v1 S1:v1
        vopc 0b5 11 v_cmpx_ne_i16_e32 S0:v1 S1:v1
        vopc 0b6 11 v_cmpx_ge_i16_e32 S0:v1 S1:v1
        vopc 0b9 11 v_cmpx_lt_u16_e32 S0:v1 S1:v1
        vopc 0ba 11 v_cmpx_eq_u16_e32 S0:v1 S1:v1
        vopc 0bb 11 v_cmpx_le_u16_e32 S0:v1 S1:v1
        vopc 0bc 11 v_cmpx_gt_u16_e32 S0:v1 S1:v1
        vopc 0bd 11 v_cmpx_ne_u16_e32 S0:v1 S1:v1
        vopc 0be 11 v_cmpx_ge_u16_e32 S0:v1 S1:v1
        vopc 0c0 11 v_cmpx_f_i32_e32 S0:v1 S1:v1
        vopc 0c1 11 v_cmpx_lt_i32_e32 S0:v1 S1:v1
        vopc 0c2 11 v_cmpx_eq_i32_e32 S0:v1 S1:v1
        vopc 0c3 11 v_cmpx_le_i32_e32 S0:v1 S1:v1
        vopc 0c4 11 v_cmpx_gt_i32_e32 S0:v1 S1:v1
        vopc 0c5 11 v_cmpx_ne_i32_e32 S0:v1 S1:v1
        vopc 0c6 11 v_cmpx_ge_i32_e32 S0:v1 S1:v1
        vopc 0c7 11 v_cmpx_t_i32_e32 S0:v1 S1:v1
        vopc 0c8 11 v_cmpx_f_u32_e32 S0:v1 S1:v1
        vopc 0c9 11 v_cmpx_lt_u32_e32 S0:v1 S1:v1
        vopc 0ca 11 v_cmpx_eq_u32_e32 S0:v1 S1:v1
        vopc 0cb 11 v_cmpx_le_u32_e32 S0:v1 S1:v1
        vopc 0cc 11 v_cmpx_gt_u32_e32 S0:v1 S1:v1
        vopc 0cd 11 v_cmpx_ne_u32_e32 S0:v1 S1:v1
        vopc 0ce 11 v_cmpx_ge_u32_e32 S0:v1 S1:v1
        vopc 0cf 11 v_cmpx_t_u32_e32 S0:v1 S1:v1
        vopc 0d0 11 v_cmpx_f_i64_e32 S0:v2 S1:v2
        vopc 0d1 11 v_cmpx_lt_i64_e32 S0:v2 S1:v2
        vopc 0d2 11 v_cmpx_eq_i64_e32 S0:v2 S1:v2
        vopc 0d3 11 v_cmpx_le_i64_e32 S0:v2 S1:v2
        vopc 0d4 11 v_cmpx_gt_i64_e32 S0:v2 S1:v2
        vopc 0d5 11 v_cmpx_ne_i64_e32 S0:v2 S1:v2
        vopc 0d6 11 v_cmpx_ge_i64_e32 S0:v2 S1:v2
        vopc 0d7 11 v_cmpx_t_i64_e32 S0:v2 S1:v2
        vopc 0d8 11 v_cmpx_f_u64_e32 S0:v2 S1:v2
        vopc 0d9 11 v_cmpx_lt_u64_e32 S0:v2 S1:v2
        vopc 0da 11 v_cmpx_eq_u64_e32 S0:v2 S1:v2
        vopc 0db 11 v_cmpx_le_u64_e32 S0:v2 S1:v2
        vopc 0dc 11 v_cmpx_gt_u64_e32 S0:v2 S1:v2
        vopc 0dd 11 v_cmpx_ne_u64_e32 S0:v2 S1:v2
        vopc 0de 11 v_cmpx_ge_u64_e32 S0:v2 S1:v2
        vopc 0df 11 v_cmpx_t_u64_e32 S0:v2 S1:v2
        vopc 0fd 11 v_cmpx_class_f16_e32 S0:v1 S1:v1
        vopc 0fe 11 v_cmpx_class_f32_e32 S0:v1 S1:v1
        vopc 0ff 11 v_cmpx_class_f64_e32 S0:v2 S1:v1
        vop3 000 11 v_cmp_f_f16_e64 D:s2 S0:v1 S1:v1
        vop3 001 11 v_cmp_lt_f16_e64 D:s2 S0:v1 S1:v1
        vop3 002 11 v_cmp_eq_f16_e64 D:s2 S0:v1 S1:v1
        vop3 003 11 v_cmp_le_f16_e64 D:s2 S0:v1 S1:v1
        vop3 004 11 v_cmp_gt_f16_e64 D:s2 S0:v1 S1:v1
        vop3 005 11 v_cmp_lg_f16_e64 D:s2 S0:v1 S1:v1
        vop3 006 11 v_cmp_ge_f16_e64 D:s2 S0:v1 S1:v1
        vop3 007 11 v_cmp_o_f16_e64 D:s2 S0:v1 S1:v1
        vop3 008 11 v_cmp_u_f16_e64 D:s2 S0:v1 S1:v1
        vop3 009 11 v_cmp_nge_f16_e64 D:s2 S0:v1 S1:v1
        vop3 00a 11 v_cmp_nlg_f16_e64 D:s2 S0:v1 S1:v1
        vop3 00b 11 v_cmp_ngt_f16_e64 D:s2 S0:v1 S1:v1
        vop3 00c 11 v_cmp_nle_f16_e64 D:s2 S0:v1 S1:v1
        vop3 00d 11 v_cmp_neq_f16_e64 D:s2 S0:v1 S1:v1
        vop3 00e 11 v_cmp_nlt_f16_e64 D:s2 S0:v1 S1:v1
        vop3 00f 11 v_cmp_t_f16_e64 D:s2 S0:v1 S1:v1
        vop3 010 11 v_cmp_f_f32_e64 D:s2 S0:v1 S1:v1
        vop3 011 11 v_cmp_lt_f32_e64 D:s2 S0:v1 S1:v1
        vop3 012 11 v_cmp_eq_f32_e64 D:s2 S0:v1 S1:v1
        vop3 013 11 v_cmp_le_f32_e64 D:s2 S0:v1 S1:v1
        vop3 014 11 v_cmp_gt_f32_e64 D:s2 S0:v1 S1:v1
        vop3 015 11 v_cmp_lg_f32_e64 D:s2 S0:v1 S1:v1
        vop3 016 11 v_cmp_ge_f32_e64 D:s2 S0:v1 S1:v1
        vop3 017 11 v_cmp_o_f32_e64 D:s2 S0:v1 S1:v1
        vop3 018 11 v_cmp_u_f32_e64 D:s2 S0:v1 S1:v1
        vop3 019 11 v_cmp_nge_f32_e64 D:s2 S0:v1 S1:v1
        vop3 01a 11 v_cmp_nlg_f32_e64 D:s2 S0:v1 S1:v1
        vop3 01b 11 v_cmp_ngt_f32_e64 D:s2 S0:v1 S1:v1
        vop3 01c 11 v_cmp_nle_f32_e64 D:s2 S0:v1 S1:v1
        vop3 01d 11 v_cmp_neq_f32_e64 D:s2 S0:v1 S1:v1
        vop3 01e 11 v_cmp_nlt_f32_e64 D:s2 S0:v1 S1:v1
        vop3 01f 11 v_cmp_t_f32_e64 D:s2 S0:v1 S1:v1
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
        vop3 02f 11 v_cmp_t_f64_e64 D:s2 S0:v2 S1:v2
        vop3 031 11 v_cmp_lt_i16_e64 D:s2 S0:v1 S1:v1
        vop3 032 11 v_cmp_eq_i16_e64 D:s2 S0:v1 S1:v1
        vop3 033 11 v_cmp_le_i16_e64 D:s2 S0:v1 S1:v1
        vop3 034 11 v_cmp_gt_i16_e64 D:s2 S0:v1 S1:v1
        vop3 035 11 v_cmp_ne_i16_e64 D:s2 S0:v1 S1:v1
        vop3 036 11 v_cmp_ge_i16_e64 D:s2 S0:v1 S1:v1
        vop3 039 11 v_cmp_lt_u16_e64 D:s2 S0:v1 S1:v1
        vop3 03a 11 v_cmp_eq_u16_e64 D:s2 S0:v1 S1:v1
        vop3 03b 11 v_cmp_le_u16_e64 D:s2 S0:v1 S1:v1
        vop3 03c 11 v_cmp_gt_u16_e64 D:s2 S0:v1 S1:v1
        vop3 03d 11 v_cmp_ne_u16_e64 D:s2 S0:v1 S1:v1
        vop3 03e 11 v_cmp_ge_u16_e64 D:s2 S0:v1 S1:v1
        vop3 040 11 v_cmp_f_i32_e64 D:s2 S0:v1 S1:v1
        vop3 041 11 v_cmp_lt_i32_e64 D:s2 S0:v1 S1:v1
        vop3 042 11 v_cmp_eq_i32_e64 D:s2 S0:v1 S1:v1
        vop3 043 11 v_cmp_le_i32_e64 D:s2 S0:v1 S1:v1
        vop3 044 11 v_cmp_gt_i32_e64 D:s2 S0:v1 S1:v1
        vop3 045 11 v_cmp_ne_i32_e64 D:s2 S0:v1 S1:v1
        vop3 046 11 v_cmp_ge_i32_e64 D:s2 S0:v1 S1:v1
        vop3 047 11 v_cmp_t_i32_e64 D:s2 S0:v1 S1:v1
        vop3 048 11 v_cmp_f_u32_e64 D:s2 S0:v1 S1:v1
        vop3 049 11 v_cmp_lt_u32_e64 D:s2 S0:v1 S1:v1
        vop3 04a 11 v_cmp_eq_u32_e64 D:s2 S0:v1 S1:v1
        vop3 04b 11 v_cmp_le_u32_e64 D:s2 S0:v1 S1:v1
        vop3 04c 11 v_cmp_gt_u32_e64 D:s2 S0:v1 S1:v1
        vop3 04d 11 v_cmp_ne_u32_e64 D:s2 S0:v1 S1:v1
        vop3 04e 11 v_cmp_ge_u32_e64 D:s2 S0:v1 S1:v1
        vop3 04f 11 v_cmp_t_u32_e64 D:s2 S0:v1 S1:v1
        vop3 050 11 v_cmp_f_i64_e64 D:s2 S0:v2 S1:v2
        vop3 051 11 v_cmp_lt_i64_e64 D:s2 S0:v2 S1:v2
        vop3 052 11 v_cmp_eq_i64_e64 D:s2 S0:v2 S1:v2
        vop3 053 11 v_cmp_le_i64_e64 D:s2 S0:v2 S1:v2
        vop3 054 11 v_cmp_gt_i64_e64 D:s2 S0:v2 S1:v2
        vop3 055 11 v_cmp_ne_i64_e64 D:s2 S0:v2 S1:v2
        vop3 056 11 v_cmp_ge_i64_e64 D:s2 S0:v2 S1:v2
        vop3 057 11 v_cmp_t_i64_e64 D:s2 S0:v2 S1:v2
        vop3 058 11 v_cmp_f_u64_e64 D:s2 S0:v2 S1:v2
        vop3 059 11 v_cmp_lt_u64_e64 D:s2 S0:v2 S1:v2
        vop3 05a 11 v_cmp_eq_u64_e64 D:s2 S0:v2 S1:v2
        vop3 05b 11 v_cmp_le_u64_e64 D:s2 S0:v2 S1:v2
        vop3 05c 11 v_cmp_gt_u64_e64 D:s2 S0:v2 S1:v2
        vop3 05d 11 v_cmp_ne_u64_e64 D:s2 S0:v2 S1:v2
        vop3 05e 11 v_cmp_ge_u64_e64 D:s2 S0:v2 S1:v2
        vop3 05f 11 v_cmp_t_u64_e64 D:s2 S0:v2 S1:v2
        vop3 07d 11 v_cmp_class_f16_e64 D:s2 S0:v1 S1:v1
        vop3 07e 11 v_cmp_class_f32_e64 D:s2 S0:v1 S1:v1
        vop3 07f 11 v_cmp_class_f64_e64 D:s2 S0:v2 S1:v1
        vop3 080 11 v_cmpx_f_f16_e64 S0:v1 S1:v1
        vop3 081 11 v_cmpx_lt_f16_e64 S0:v1 S1:v1
        vop3 082 11 v_cmpx_eq_f16_e64 S0:v1 S1:v1
        vop3 083 11 v_cmpx_le_f16_e64 S0:v1 S1:v1
        vop3 084 11 v_cmpx_gt_f16_e64 S0:v1 S1:v1
        vop3 085 11 v_cmpx_lg_f16_e64 S0:v1 S1:v1
        vop3 086 11 v_cmpx_ge_f16_e64 S0:v1 S1:v1
        vop3 087 11 v_cmpx_o_f16_e64 S0:v1 S1:v1
        vop3 088 11 v_cmpx_u_f16_e64 S0:v1 S1:v1
        vop3 089 11 v_cmpx_nge_f16_e64 S0:v1 S1:v1
        vop3 08a 11 v_cmpx_nlg_f16_e64 S0:v1 S1:v1
        vop3 08b 11 v_cmpx_ngt_f16_e64 S0:v1 S1:v1
        vop3 08c 11 v_cmpx_nle_f16_e64 S0:v1 S1:v1
        vop3 08d 11 v_cmpx_neq_f16_e64 S0:v1 S1:v1
        vop3 08e 11 v_cmpx_nlt_f16_e64 S0:v1 S1:v1
        vop3 08f 11 v_cmpx_t_f16_e64 S0:v1 S1:v1
        vop3 090 11 v_cmpx_f_f32_e64 S0:v1 S1:v1
        vop3 091 11 v_cmpx_lt_f32_e64 S0:v1 S1:v1
        vop3 092 11 v_cmpx_eq_f32_e64 S0:v1 S1:v1
        vop3 093 11 v_cmpx_le_f32_e64 S0:v1 S1:v1
        vop3 094 11 v_cmpx_gt_f32_e64 S0:v1 S1:v1
        vop3 095 11 v_cmpx_lg_f32_e64 S0:v1 S1:v1
        vop3 096 11 v_cmpx_ge_f32_e64 S0:v1 S1:v1
        vop3 097 11 v_cmpx_o_f32_e64 S0:v1 S1:v1
        vop3 098 11 v_cmpx_u_f32_e64 S0:v1 S1:v1
        vop3 099 11 v_cmpx_nge_f32_e64 S0:v1 S1:v1
        vop3 09a 11 v_cmpx_nlg_f32_e64 S0:v1 S1:v1
        vop3 09b 11 v_cmpx_ngt_f32_e64 S0:v1 S1:v1
        vop3 09c 11 v_cmpx_nle_f32_e64 S0:v1 S1:v1
        vop3 09d 11 v_cmpx_neq_f32_e64 S0:v1 S1:v1
        vop3 09e 11 v_cmpx_nlt_f32_e64 S0:v1 S1:v1
        vop3 09f 11 v_cmpx_t_f32_e64 S0:v1 S1:v1
        vop3 0a0 11 v_cmpx_f_f64_e64 S0:v2 S1:v2
        vop3 0a1 11 v_cmpx_lt_f64_e64 S0:v2 S1:v2
        vop3 0a2 11 v_cmpx_eq_f64_e64 S0:v2 S1:v2
        vop3 0a3 11 v_cmpx_le_f64_e64 S0:v2 S1:v2
        vop3 0a4 11 v_cmpx_gt_f64_e64 S0:v2 S1:v2
        vop3 0a5 11 v_cmpx_lg_f64_e64 S0:v2 S1:v2
        vop3 0a6 11 v_cmpx_ge_f64_e64 S0:v2 S1:v2
        vop3 0a7 11 v_cmpx_o_f64_e64 S0:v2 S1:v2
        vop3 0a8 11 v_cmpx_u_f64_e64 S0:v2 S1:v2
        vop3 0a9 11 v_cmpx_nge_f64_e64 S0:v2 S1:v2
        vop3 0aa 11 v_cmpx_nlg_f64_e64 S0:v2 S1:v2
        vop3 0ab 11 v_cmpx_ngt_f64_e64 S0:v2 S1:v2
        vop3 0ac 11 v_cmpx_nle_f64_e64 S0:v2 S1:v2
        vop3 0ad 11 v_cmpx_neq_f64_e64 S0:v2 S1:v2
        vop3 0ae 11 v_cmpx_nlt_f64_e64 S0:v2 S1:v2
        vop3 0af 11 v_cmpx_t_f64_e64 S0:v2 S1:v2
        vop3 0b1 11 v_cmpx_lt_i16_e64 S0:v1 S1:v1
        vop3 0b2 11 v_cmpx_eq_i16_e64 S0:v1 S1:v1
        vop3 0b3 11 v_cmpx_le_i16_e64 S0:v1 S1:v1
        vop3 0b4 11 v_cmpx_gt_i16_e64 S0:v1 S1:v1
        vop3 0b5 11 v_cmpx_ne_i16_e64 S0:v1 S1:v1
        vop3 0b6 11 v_cmpx_ge_i16_e64 S0:v1 S1:v1
        vop3 0b9 11 v_cmpx_lt_u16_e64 S0:v1 S1:v1
        vop3 0ba 11 v_cmpx_eq_u16_e64 S0:v1 S1:v1
        vop3 0bb 11 v_cmpx_le_u16_e64 S0:v1 S1:v1
        vop3 0bc 11 v_cmpx_gt_u16_e64 S0:v1 S1:v1
        vop3 0bd 11 v_cmpx_ne_u16_e64 S0:v1 S1:v1
        vop3 0be 11 v_cmpx_ge_u16_e64 S0:v1 S1:v1
        vop3 0c0 11 v_cmpx_f_i32_e64 S0:v1 S1:v1
        vop3 0c1 11 v_cmpx_lt_i32_e64 S0:v1 S1:v1
        vop3 0c2 11 v_cmpx_eq_i32_e64 S0:v1 S1:v1
        vop3 0c3 11 v_cmpx_le_i32_e64 S0:v1 S1:v1
        vop3 0c4 11 v_cmpx_gt_i32_e64 S0:v1 S1:v1
        vop3 0c5 11 v_cmpx_ne_i32_e64 S0:v1 S1:v1
        vop3 0c6 11 v_cmpx_ge_i32_e64 S0:v1 S1:v1
        vop3 0c7 11 v_cmpx_t_i32_e64 S0:v1 S1:v1
        vop3 0c8 11 v_cmpx_f_u32_e64 S0:v1 S1:v1
        vop3 0c9 11 v_cmpx_lt_u32_e64 S0:v1 S1:v1
        vop3 0ca 11 v_cmpx_eq_u32_e64 S0:v1 S1:v1
        vop3 0cb 11 v_cmpx_le_u32_e64 S0:v1 S1:v1
        vop3 0cc 11 v_cmpx_gt_u32_e64 S0:v1 S1:v1
        vop3 0cd 11 v_cmpx_ne_u32_e64 S0:v1 S1:v1
        vop3 0ce 11 v_cmpx_ge_u32_e64 S0:v1 S1:v1
        vop3 0cf 11 v_cmpx_t_u32_e64 S0:v1 S1:v1
        vop3 0d0 11 v_cmpx_f_i64_e64 S0:v2 S1:v2
        vop3 0d1 11 v_cmpx_lt_i64_e64 S0:v2 S1:v2
        vop3 0d2 11 v_cmpx_eq_i64_e64 S0:v2 S1:v2
        vop3 0d3 11 v_cmpx_le_i64_e64 S0:v2 S1:v2
        vop3 0d4 11 v_cmpx_gt_i64_e64 S0:v2 S1:v2
        vop3 0d5 11 v_cmpx_ne_i64_e64 S0:v2 S1:v2
        vop3 0d6 11 v_cmpx_ge_i64_e64 S0:v2 S1:v2
        vop3 0d7 11 v_cmpx_t_i64_e64 S0:v2 S1:v2
        vop3 0d8 11 v_cmpx_f_u64_e64 S0:v2 S1:v2
        vop3 0d9 11 v_cmpx_lt_u64_e64 S0:v2 S1:v2
        vop3 0da 11 v_cmpx_eq_u64_e64 S0:v2 S1:v2
        vop3 0db 11 v_cmpx_le_u64_e64 S0:v2 S1:v2
        vop3 0dc 11 v_cmpx_gt_u64_e64 S0:v2 S1:v2
        vop3 0dd 11 v_cmpx_ne_u64_e64 S0:v2 S1:v2
        vop3 0de 11 v_cmpx_ge_u64_e64 S0:v2 S1:v2
        vop3 0df 11 v_cmpx_t_u64_e64 S0:v2 S1:v2
        vop3 0fd 11 v_cmpx_class_f16_e64 S0:v1 S1:v1
        vop3 0fe 11 v_cmpx_class_f32_e64 S0:v1 S1:v1
        vop3 0ff 11 v_cmpx_class_f64_e64 S0:v2 S1:v1
        vop3 101 11 v_cndmask_b32_e64 D:v1 S0:v1 S1:v1 S2:s2
        vop3 103 11 v_add_f32_e64 D:v1 S0:v1 S1:v1
        vop3 104 11 v_sub_f32_e64 D:v1 S0:v1 S1:v1
        vop3 105 11 v_subrev_f32_e64 D:v1 S0:v1 S1:v1
        vop3 106 11 v_fmac_dx9_zero_f32_e64 D:v1 S0:v1 S1:v1
        vop3 107 11 v_mul_dx9_zero_f32_e64 D:v1 S0:v1 S1:v1
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
        vop3 118 11 v_lshlrev_b32_e64 D:v1 S0:v1 S1:v1
        vop3 119 11 v_lshrrev_b32_e64 D:v1 S0:v1 S1:v1
        vop3 11a 11 v_ashrrev_i32_e64 D:v1 S0:v1 S1:v1
        vop3 11b 11 v_and_b32_e64 D:v1 S0:v1 S1:v1
        vop3 11c 11 v_or_b32_e64 D:v1 S0:v1 S1:v1
        vop3 11d 11 v_xor_b32_e64 D:v1 S0:v1 S1:v1
        vop3 11e 11 v_xnor_b32_e64 D:v1 S0:v1 S1:v1
        vop3 120 11 v_add_co_ci_u32_e64 D:v1 SD:s2 S0:v1 S1:v1 S2:s2
        vop3 121 11 v_sub_co_ci_u32_e64 D:v1 SD:s2 S0:v1 S1:v1 S2:s2
        vop3 122 11 v_subrev_co_ci_u32_e64 D:v1 SD:s2 S0:v1 S1:v1 S2:s2
        vop3 125 11 v_add_nc_u32_e64 D:v1 S0:v1 S1:v1
        vop3 126 11 v_sub_nc_u32_e64 D:v1 S0:v1 S1:v1
        vop3 127 11 v_subrev_nc_u32_e64 D:v1 S0:v1 S1:v1
        vop3 12b 11 v_fmac_f32_e64 D:v1 S0:v1 S1:v1
        vop3 12f 11 v_cvt_pk_rtz_f16_f32_e64 D:v1 S0:v1 S1:v1
        vop3 132 11 v_add_f16_e64 D:v1 S0:v1 S1:v1
        vop3 133 11 v_sub_f16_e64 D:v1 S0:v1 S1:v1
        vop3 134 11 v_subrev_f16_e64 D:v1 S0:v1 S1:v1
        vop3 135 11 v_mul_f16_e64 D:v1 S0:v1 S1:v1
        vop3 136 11 v_fmac_f16_e64 D:v1 S0:v1 S1:v1
        vop3 139 11 v_max_f16_e64 D:v1 S0:v1 S1:v1
        vop3 13a 11 v_min_f16_e64 D:v1 S0:v1 S1:v1
        vop3 13b 11 v_ldexp_f16_e64 D:v1 S0:v1 S1:v1
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
        vop3 18c 11 v_cvt_nearest_i32_f32_e64 D:v1 S0:v1
        vop3 18d 11 v_cvt_floor_i32_f32_e64 D:v1 S0:v1
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
        vop3 1b9 11 v_clz_i32_u32_e64 D:v1 S0:v1
        vop3 1ba 11 v_ctz_i32_b32_e64 D:v1 S0:v1
        vop3 1bb 11 v_cls_i32_e64 D:v1 S0:v1
        vop3 1bc 11 v_frexp_exp_i32_f64_e64 D:v1 S0:v2
        vop3 1bd 11 v_frexp_mant_f64_e64 D:v2 S0:v2
        vop3 1be 11 v_fract_f64_e64 D:v2 S0:v2
        vop3 1bf 11 v_frexp_exp_i32_f32_e64 D:v1 S0:v1
        vop3 1c0 11 v_frexp_mant_f32_e64 D:v1 S0:v1
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
        vop3 1e9 11 v_not_b16_e64 D:v1 S0:v1
        vop3 1ea 11 v_cvt_i32_i16_e64 D:v1 S0:v1
        vop3 1eb 11 v_cvt_u32_u16_e64 D:v1 S0:v1
        vop3 209 11 v_fma_dx9_zero_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 20a 11 v_mad_i32_i24 D:v1 S0:v1 S1:v1 S2:v1
        vop3 20b 11 v_mad_u32_u24 D:v1 S0:v1 S1:v1 S2:v1
        vop3 20c 11 v_cubeid_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 20d 11 v_cubesc_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 20e 11 v_cubetc_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 20f 11 v_cubema_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 210 11 v_bfe_u32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 211 11 v_bfe_i32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 212 11 v_bfi_b32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 213 11 v_fma_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 214 11 v_fma_f64 D:v2 S0:v2 S1:v2 S2:v2
        vop3 215 11 v_lerp_u8 D:v1 S0:v1 S1:v1 S2:v1
        vop3 216 11 v_alignbit_b32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 217 11 v_alignbyte_b32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 218 11 v_mullit_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 219 11 v_min3_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 21a 11 v_min3_i32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 21b 11 v_min3_u32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 21c 11 v_max3_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 21d 11 v_max3_i32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 21e 11 v_max3_u32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 21f 11 v_med3_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 220 11 v_med3_i32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 221 11 v_med3_u32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 222 11 v_sad_u8 D:v1 S0:v1 S1:v1 S2:v1
        vop3 223 11 v_sad_hi_u8 D:v1 S0:v1 S1:v1 S2:v1
        vop3 224 11 v_sad_u16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 225 11 v_sad_u32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 226 11 v_cvt_pk_u8_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 227 11 v_div_fixup_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 228 11 v_div_fixup_f64 D:v2 S0:v2 S1:v2 S2:v2
        vop3 237 11 v_div_fmas_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 238 11 v_div_fmas_f64 D:v2 S0:v2 S1:v2 S2:v2
        vop3 239 11 v_msad_u8 D:v1 S0:v1 S1:v1 S2:v1
        vop3 23a 11 v_qsad_pk_u16_u8 D:v2 S0:v2 S1:v1 S2:v2
        vop3 23b 11 v_mqsad_pk_u16_u8 D:v2 S0:v2 S1:v1 S2:v2
        vop3 23d 11 v_mqsad_u32_u8 D:v4 S0:v2 S1:v1 S2:v4
        vop3 240 11 v_xor3_b32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 241 11 v_mad_u16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 244 11 v_perm_b32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 245 11 v_xad_u32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 246 11 v_lshl_add_u32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 247 11 v_add_lshl_u32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 248 11 v_fma_f16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 249 11 v_min3_f16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 24a 11 v_min3_i16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 24b 11 v_min3_u16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 24c 11 v_max3_f16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 24d 11 v_max3_i16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 24e 11 v_max3_u16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 24f 11 v_med3_f16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 250 11 v_med3_i16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 251 11 v_med3_u16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 253 11 v_mad_i16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 254 11 v_div_fixup_f16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 255 11 v_add3_u32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 256 11 v_lshl_or_b32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 257 11 v_and_or_b32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 258 11 v_or3_b32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 259 11 v_mad_u32_u16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 25a 11 v_mad_i32_i16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 25b 11 v_permlane16_b32 D:v1 S0:v1 S1:s1 S2:s1
        vop3 25c 11 v_permlanex16_b32 D:v1 S0:v1 S1:s1 S2:s1
        vop3 25d 11 v_cndmask_b16 D:v1 S0:v1 S1:v1 S2:s2
        vop3 25e 11 v_maxmin_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 25f 11 v_minmax_f32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 260 11 v_maxmin_f16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 261 11 v_minmax_f16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 262 11 v_maxmin_u32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 263 11 v_minmax_u32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 264 11 v_maxmin_i32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 265 11 v_minmax_i32 D:v1 S0:v1 S1:v1 S2:v1
        vop3 266 11 v_dot2_f16_f16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 267 11 v_dot2_bf16_bf16 D:v1 S0:v1 S1:v1 S2:v1
        vop3 2fc 11 v_div_scale_f32 D:v1 SD:s2 S0:v1 S1:v1 S2:v1
        vop3 2fd 11 v_div_scale_f64 D:v2 SD:s2 S0:v2 S1:v2 S2:v2
        vop3 2fe 11 v_mad_u64_u32 D:v2 SD:s2 S0:v1 S1:v1 S2:v2
        vop3 2ff 11 v_mad_i64_i32 D:v2 SD:s2 S0:v1 S1:v1 S2:v2
        vop3 300 11 v_add_co_u32 D:v1 SD:s2 S0:v1 S1:v1
        vop3 301 11 v_sub_co_u32 D:v1 SD:s2 S0:v1 S1:v1
        vop3 302 11 v_subrev_co_u32 D:v1 SD:s2 S0:v1 S1:v1
        vop3 303 11 v_add_nc_u16 D:v1 S0:v1 S1:v1
        vop3 304 11 v_sub_nc_u16 D:v1 S0:v1 S1:v1
        vop3 305 11 v_mul_lo_u16 D:v1 S0:v1 S1:v1
        vop3 306 11 v_cvt_pk_i16_f32 D:v1 S0:v1 S1:v1
        vop3 307 11 v_cvt_pk_u16_f32 D:v1 S0:v1 S1:v1
        vop3 309 11 v_max_u16 D:v1 S0:v1 S1:v1
        vop3 30a 11 v_max_i16 D:v1 S0:v1 S1:v1
        vop3 30b 11 v_min_u16 D:v1 S0:v1 S1:v1
        vop3 30c 11 v_min_i16 D:v1 S0:v1 S1:v1
        vop3 30d 11 v_add_nc_i16 D:v1 S0:v1 S1:v1
        vop3 30e 11 v_sub_nc_i16 D:v1 S0:v1 S1:v1
        vop3 311 11 v_pack_b32_f16 D:v1 S0:v1 S1:v1
        vop3 312 11 v_cvt_pk_norm_i16_f16 D:v1 S0:v1 S1:v1
        vop3 313 11 v_cvt_pk_norm_u16_f16 D:v1 S0:v1 S1:v1
        vop3 31c 11 v_ldexp_f32 D:v1 S0:v1 S1:v1
        vop3 31d 11 v_bfm_b32 D:v1 S0:v1 S1:v1
        vop3 31e 11 v_bcnt_u32_b32 D:v1 S0:v1 S1:v1
        vop3 31f 11 v_mbcnt_lo_u32_b32 D:v1 S0:v1 S1:v1
        vop3 320 11 v_mbcnt_hi_u32_b32 D:v1 S0:v1 S1:v1
        vop3 321 11 v_cvt_pk_norm_i16_f32 D:v1 S0:v1 S1:v1
        vop3 322 11 v_cvt_pk_norm_u16_f32 D:v1 S0:v1 S1:v1
        vop3 323 11 v_cvt_pk_u16_u32 D:v1 S0:v1 S1:v1
        vop3 324 11 v_cvt_pk_i16_i32 D:v1 S0:v1 S1:v1
        vop3 325 11 v_sub_nc_i32 D:v1 S0:v1 S1:v1
        vop3 326 11 v_add_nc_i32 D:v1 S0:v1 S1:v1
        vop3 327 11 v_add_f64 D:v2 S0:v2 S1:v2
        vop3 328 11 v_mul_f64 D:v2 S0:v2 S1:v2
        vop3 329 11 v_min_f64 D:v2 S0:v2 S1:v2
        vop3 32a 11 v_max_f64 D:v2 S0:v2 S1:v2
        vop3 32b 11 v_ldexp_f64 D:v2 S0:v2 S1:v1
        vop3 32c 11 v_mul_lo_u32 D:v1 S0:v1 S1:v1
        vop3 32d 11 v_mul_hi_u32 D:v1 S0:v1 S1:v1
        vop3 32e 11 v_mul_hi_i32 D:v1 S0:v1 S1:v1
        vop3 32f 11 v_trig_preop_f64 D:v2 S0:v2 S1:v1
        vop3 338 11 v_lshlrev_b16 D:v1 S0:v1 S1:v1
        vop3 339 11 v_lshrrev_b16 D:v1 S0:v1 S1:v1
        vop3 33a 11 v_ashrrev_i16 D:v1 S0:v1 S1:v1
        vop3 33c 11 v_lshlrev_b64 D:v2 S0:v1 S1:v2
        vop3 33d 11 v_lshrrev_b64 D:v2 S0:v1 S1:v2
        vop3 33e 11 v_ashrrev_i64 D:v2 S0:v1 S1:v2
        vop3 360 11 v_readlane_b32 D:s1 S0:v1 S1:s1
        vop3 361 11 v_writelane_b32 D:v1 S0:s1 S1:s1
        vop3 362 11 v_and_b16 D:v1 S0:v1 S1:v1
        vop3 363 11 v_or_b16 D:v1 S0:v1 S1:v1
        vop3 364 11 v_xor_b16 D:v1 S0:v1 S1:v1
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
        vop3p 013 11 v_dot2_f32_f16 D:v1 S0:v1 S1:v1 S2:v1
        vop3p 016 11 v_dot4_i32_iu8 D:v1 S0:v1 S1:v1 S2:v1
        vop3p 017 11 v_dot4_u32_u8 D:v1 S0:v1 S1:v1 S2:v1
        vop3p 018 11 v_dot8_i32_iu4 D:v1 S0:v1 S1:v1 S2:v1
        vop3p 019 11 v_dot8_u32_u4 D:v1 S0:v1 S1:v1 S2:v1
        vop3p 01a 11 v_dot2_f32_bf16 D:v1 S0:v1 S1:v1 S2:v1
        vop3p 020 11 v_fma_mix_f32 D:v1 S0:v1 S1:v1 S2:v1 mix
        vop3p 021 11 v_fma_mixlo_f16 D:v1 S0:v1 S1:v1 S2:v1 mix
        vop3p 022 11 v_fma_mixhi_f16 D:v1 S0:v1 S1:v1 S2:v1 mix
        vop3p 040 11 v_wmma_f32_16x16x16_f16 D:v8 S0:v8 S1:v8 S2:v8
        vop3p 041 11 v_wmma_f32_16x16x16_bf16 D:v8 S0:v8 S1:v8 S2:v8
        vop3p 042 11 v_wmma_f16_16x16x16_f16 D:v8 S0:v8 S1:v8 S2:v8
        vop3p 043 11 v_wmma_bf16_16x16x16_bf16 D:v8 S0:v8 S1:v8 S2:v8
        vop3p 044 11 v_wmma_i32_16x16x16_iu8 D:v8 S0:v4 S1:v4 S2:v8
        vop3p 045 11 v_wmma_i32_16x16x16_iu4 D:v8 S0:v2 S1:v2 S2:v8
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
        ds 00d 11 ds_store_b32 A:v1 S0:v1 O:16
        ds 00e 11 ds_store_2addr_b32 A:v1 S0:v1 S1:v1 O:8x2
        ds 00f 11 ds_store_2addr_stride64_b32 A:v1 S0:v1 S1:v1 O:8x2
        ds 010 11 ds_cmpstore_b32 A:v1 S0:v1 S1:v1 O:16
        ds 011 11 ds_cmpstore_f32 A:v1 S0:v1 S1:v1 O:16
        ds 012 11 ds_min_f32 A:v1 S0:v1 O:16
        ds 013 11 ds_max_f32 A:v1 S0:v1 O:16
        ds 015 11 ds_add_f32 A:v1 S0:v1 O:16
        ds 01e 11 ds_store_b8 A:v1 S0:v1 O:16
        ds 01f 11 ds_store_b16 A:v1 S0:v1 O:16
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
        ds 02d 11 ds_storexchg_rtn_b32 D:v1 A:v1 S0:v1 O:16
        ds 02e 11 ds_storexchg_2addr_rtn_b32 D:v2 A:v1 S0:v1 S1:v1 O:8x2
        ds 02f 11 ds_storexchg_2addr_stride64_rtn_b32 D:v2 A:v1 S0:v1 S1:v1 O:8x2
        ds 030 11 ds_cmpstore_rtn_b32 D:v1 A:v1 S0:v1 S1:v1 O:16
        ds 031 11 ds_cmpstore_rtn_f32 D:v1 A:v1 S0:v1 S1:v1 O:16
        ds 032 11 ds_min_rtn_f32 D:v1 A:v1 S0:v1 O:16
        ds 033 11 ds_max_rtn_f32 D:v1 A:v1 S0:v1 O:16
        ds 034 11 ds_wrap_rtn_b32 D:v1 A:v1 S0:v1 S1:v1 O:16
        ds 035 11 ds_swizzle_b32 D:v1 A:v1 O:swz
        ds 036 11 ds_load_b32 D:v1 A:v1 O:16
        ds 037 11 ds_load_2addr_b32 D:v2 A:v1 O:8x2
        ds 038 11 ds_load_2addr_stride64_b32 D:v2 A:v1 O:8x2
        ds 039 11 ds_load_i8 D:v1 A:v1 O:16
        ds 03a 11 ds_load_u8 D:v1 A:v1 O:16
        ds 03b 11 ds_load_i16 D:v1 A:v1 O:16
        ds 03c 11 ds_load_u16 D:v1 A:v1 O:16
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
        ds 04d 11 ds_store_b64 A:v1 S0:v2 O:16
        ds 04e 11 ds_store_2addr_b64 A:v1 S0:v2 S1:v2 O:8x2
        ds 04f 11 ds_store_2addr_stride64_b64 A:v1 S0:v2 S1:v2 O:8x2
        ds 050 11 ds_cmpstore_b64 A:v1 S0:v2 S1:v2 O:16
        ds 051 11 ds_cmpstore_f64 A:v1 S0:v2 S1:v2 O:16
        ds 052 11 ds_min_f64 A:v1 S0:v2 O:16
        ds 053 11 ds_max_f64 A:v1 S0:v2 O:16
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
        ds 06d 11 ds_storexchg_rtn_b64 D:v2 A:v1 S0:v2 O:16
        ds 06e 11 ds_storexchg_2addr_rtn_b64 D:v4 A:v1 S0:v2 S1:v2 O:8x2
        ds 06f 11 ds_storexchg_2addr_stride64_rtn_b64 D:v4 A:v1 S0:v2 S1:v2 O:8x2
        ds 070 11 ds_cmpstore_rtn_b64 D:v2 A:v1 S0:v2 S1:v2 O:16
        ds 071 11 ds_cmpstore_rtn_f64 D:v2 A:v1 S0:v2 S1:v2 O:16
        ds 072 11 ds_min_rtn_f64 D:v2 A:v1 S0:v2 O:16
        ds 073 11 ds_max_rtn_f64 D:v2 A:v1 S0:v2 O:16
        ds 076 11 ds_load_b64 D:v2 A:v1 O:16
        ds 077 11 ds_load_2addr_b64 D:v4 A:v1 O:8x2
        ds 078 11 ds_load_2addr_stride64_b64 D:v4 A:v1 O:8x2
        ds 079 11 ds_add_rtn_f32 D:v1 A:v1 S0:v1 O:16
        ds 07e 11 ds_condxchg32_rtn_b64 D:v2 A:v1 S0:v2 O:16
        ds 0a0 11 ds_store_b8_d16_hi A:v1 S0:v1 O:16
        ds 0a1 11 ds_store_b16_d16_hi A:v1 S0:v1 O:16
        ds 0a2 11 ds_load_u8_d16 D:v1 A:v1 O:16
        ds 0a3 11 ds_load_u8_d16_hi D:v1 A:v1 O:16
        ds 0a4 11 ds_load_i8_d16 D:v1 A:v1 O:16
        ds 0a5 11 ds_load_i8_d16_hi D:v1 A:v1 O:16
        ds 0a6 11 ds_load_u16_d16 D:v1 A:v1 O:16
        ds 0a7 11 ds_load_u16_d16_hi D:v1 A:v1 O:16
        ds 0ad 11 ds_bvh_stack_rtn_b32 D:v1 A:v1 S0:v1 S1:v4 O:16
        ds 0b0 11 ds_store_addtid_b32 S0:v1 O:16
        ds 0b1 11 ds_load_addtid_b32 D:v1 O:16
        ds 0b2 11 ds_permute_b32 D:v1 A:v1 S0:v1 O:16
        ds 0b3 11 ds_bpermute_b32 D:v1 A:v1 S0:v1 O:16
        ds 0de 11 ds_store_b96 A:v1 S0:v3 O:16
        ds 0df 11 ds_store_b128 A:v1 S0:v4 O:16
        ds 0fe 11 ds_load_b96 D:v3 A:v1 O:16
        ds 0ff 11 ds_load_b128 D:v4 A:v1 O:16
        flat 010 11 flat_load_u8 D:v1 A:v2 O:16
        flat 011 11 flat_load_i8 D:v1 A:v2 O:16
        flat 012 11 flat_load_u16 D:v1 A:v2 O:16
        flat 013 11 flat_load_i16 D:v1 A:v2 O:16
        flat 014 11 flat_load_b32 D:v1 A:v2 O:16
        flat 015 11 flat_load_b64 D:v2 A:v2 O:16
        flat 016 11 flat_load_b96 D:v3 A:v2 O:16
        flat 017 11 flat_load_b128 D:v4 A:v2 O:16
        flat 018 11 flat_store_b8 A:v2 S0:v1 O:16
        flat 019 11 flat_store_b16 A:v2 S0:v1 O:16
        flat 01a 11 flat_store_b32 A:v2 S0:v1 O:16
        flat 01b 11 flat_store_b64 A:v2 S0:v2 O:16
        flat 01c 11 flat_store_b96 A:v2 S0:v3 O:16
        flat 01d 11 flat_store_b128 A:v2 S0:v4 O:16
        flat 01e 11 flat_load_d16_u8 D:v1 A:v2 O:16
        flat 01f 11 flat_load_d16_i8 D:v1 A:v2 O:16
        flat 020 11 flat_load_d16_b16 D:v1 A:v2 O:16
        flat 021 11 flat_load_d16_hi_u8 D:v1 A:v2 O:16
        flat 022 11 flat_load_d16_hi_i8 D:v1 A:v2 O:16
        flat 023 11 flat_load_d16_hi_b16 D:v1 A:v2 O:16
        flat 024 11 flat_store_d16_hi_b8 A:v2 S0:v1 O:16
        flat 025 11 flat_store_d16_hi_b16 A:v2 S0:v1 O:16
        flat 033 11 flat_atomic_swap_b32 A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 034 11 flat_atomic_cmpswap_b32 A:v2 S0:v2 O:16 | D:v1 A:v2 S0:v2 O:16
        flat 035 11 flat_atomic_add_u32 A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 036 11 flat_atomic_sub_u32 A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 038 11 flat_atomic_min_i32 A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 039 11 flat_atomic_min_u32 A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 03a 11 flat_atomic_max_i32 A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 03b 11 flat_atomic_max_u32 A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 03c 11 flat_atomic_and_b32 A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 03d 11 flat_atomic_or_b32 A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 03e 11 flat_atomic_xor_b32 A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 03f 11 flat_atomic_inc_u32 A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 040 11 flat_atomic_dec_u32 A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 041 11 flat_atomic_swap_b64 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 042 11 flat_atomic_cmpswap_b64 A:v2 S0:v4 O:16 | D:v2 A:v2 S0:v4 O:16
        flat 043 11 flat_atomic_add_u64 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 044 11 flat_atomic_sub_u64 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 045 11 flat_atomic_min_i64 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 046 11 flat_atomic_min_u64 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 047 11 flat_atomic_max_i64 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 048 11 flat_atomic_max_u64 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 049 11 flat_atomic_and_b64 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 04a 11 flat_atomic_or_b64 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 04b 11 flat_atomic_xor_b64 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 04c 11 flat_atomic_inc_u64 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 04d 11 flat_atomic_dec_u64 A:v2 S0:v2 O:16 | D:v2 A:v2 S0:v2 O:16
        flat 050 11 flat_atomic_cmpswap_f32 A:v2 S0:v2 O:16 | D:v1 A:v2 S0:v2 O:16
        flat 051 11 flat_atomic_min_f32 A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 052 11 flat_atomic_max_f32 A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        flat 056 11 flat_atomic_add_f32 A:v2 S0:v1 O:16 | D:v1 A:v2 S0:v1 O:16
        global 010 11 global_load_u8 D:v1 A:v2 =off O:16
        global 011 11 global_load_i8 D:v1 A:v2 =off O:16
        global 012 11 global_load_u16 D:v1 A:v2 =off O:16
        global 013 11 global_load_i16 D:v1 A:v2 =off O:16
        global 014 11 global_load_b32 D:v1 A:v2 =off O:16
        global 015 11 global_load_b64 D:v2 A:v2 =off O:16
        global 016 11 global_load_b96 D:v3 A:v2 =off O:16
        global 017 11 global_load_b128 D:v4 A:v2 =off O:16
        global 018 11 global_store_b8 A:v2 S0:v1 =off O:16
        global 019 11 global_store_b16 A:v2 S0:v1 =off O:16
        global 01a 11 global_store_b32 A:v2 S0:v1 =off O:16
        global 01b 11 global_store_b64 A:v2 S0:v2 =off O:16
        global 01c 11 global_store_b96 A:v2 S0:v3 =off O:16
        global 01d 11 global_store_b128 A:v2 S0:v4 =off O:16
        global 01e 11 global_load_d16_u8 D:v1 A:v2 =off O:16
        global 01f 11 global_load_d16_i8 D:v1 A:v2 =off O:16
        global 020 11 global_load_d16_b16 D:v1 A:v2 =off O:16
        global 021 11 global_load_d16_hi_u8 D:v1 A:v2 =off O:16
        global 022 11 global_load_d16_hi_i8 D:v1 A:v2 =off O:16
        global 023 11 global_load_d16_hi_b16 D:v1 A:v2 =off O:16
        global 024 11 global_store_d16_hi_b8 A:v2 S0:v1 =off O:16
        global 025 11 global_store_d16_hi_b16 A:v2 S0:v1 =off O:16
        global 028 11 global_load_addtid_b32 D:v1 =off O:16
        global 029 11 global_store_addtid_b32 S0:v1 =off O:16
        global 033 11 global_atomic_swap_b32 A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 034 11 global_atomic_cmpswap_b32 A:v2 S0:v2 =off O:16 | D:v1 A:v2 S0:v2 =off O:16
        global 035 11 global_atomic_add_u32 A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 036 11 global_atomic_sub_u32 A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 037 11 global_atomic_csub_u32 A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 038 11 global_atomic_min_i32 A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 039 11 global_atomic_min_u32 A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 03a 11 global_atomic_max_i32 A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 03b 11 global_atomic_max_u32 A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 03c 11 global_atomic_and_b32 A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 03d 11 global_atomic_or_b32 A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 03e 11 global_atomic_xor_b32 A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 03f 11 global_atomic_inc_u32 A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 040 11 global_atomic_dec_u32 A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 041 11 global_atomic_swap_b64 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 042 11 global_atomic_cmpswap_b64 A:v2 S0:v4 =off O:16 | D:v2 A:v2 S0:v4 =off O:16
        global 043 11 global_atomic_add_u64 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 044 11 global_atomic_sub_u64 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 045 11 global_atomic_min_i64 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 046 11 global_atomic_min_u64 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 047 11 global_atomic_max_i64 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 048 11 global_atomic_max_u64 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 049 11 global_atomic_and_b64 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 04a 11 global_atomic_or_b64 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 04b 11 global_atomic_xor_b64 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 04c 11 global_atomic_inc_u64 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 04d 11 global_atomic_dec_u64 A:v2 S0:v2 =off O:16 | D:v2 A:v2 S0:v2 =off O:16
        global 050 11 global_atomic_cmpswap_f32 A:v2 S0:v2 =off O:16 | D:v1 A:v2 S0:v2 =off O:16
        global 051 11 global_atomic_min_f32 A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 052 11 global_atomic_max_f32 A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        global 056 11 global_atomic_add_f32 A:v2 S0:v1 =off O:16 | D:v1 A:v2 S0:v1 =off O:16
        mubuf 02b 11 buffer_gl0_inv -
        mubuf 02c 11 buffer_gl1_inv -
        """;
}
