; The gfx9 instruction forms Idrak's GCN kernels use (GCN 5 and CDNA), one per line, for the goldens in forms.golden.
; tools/gcn/make_gcn_golden.py assembles this file with llvm-mc for each gfx9 CPU; a CPU that rejects a line lists it
; in forms.golden's "# rejected:" section for that CPU. Labels are resolved within the whole file, assembled as one
; block from offset 0. Comments start with ';'. Registers that hold 64-bit or wider values are even-aligned (gfx90a
; and later require it); the one unaligned pair is marked.

; ===================================================================================================================
; SOP1: one scalar source
; ===================================================================================================================
  s_mov_b32 s0, s1
  s_mov_b32 s0, 0
  s_mov_b32 s0, 64
  s_mov_b32 s0, -1
  s_mov_b32 s0, -16
  s_mov_b32 s0, 1.0
  s_mov_b32 s0, -0.5
  s_mov_b32 s0, 0.15915494
  s_mov_b32 s0, 65
  s_mov_b32 s0, 0x12345678
  s_mov_b32 s0, 0x3fb8aa3b
  s_mov_b32 s101, s0
  s_mov_b32 s0, vcc_lo
  s_mov_b32 s0, exec_hi
  s_mov_b32 m0, s2
  s_mov_b32 m0, -1
  s_mov_b64 s[0:1], s[2:3]
  s_mov_b64 s[0:1], 0
  s_mov_b64 s[0:1], -1
  s_mov_b64 s[0:1], exec
  s_mov_b64 s[0:1], 0x12345678
  s_cmov_b32 s0, s1
  s_cmov_b64 s[0:1], s[2:3]
  s_not_b32 s0, s1
  s_not_b64 s[0:1], s[2:3]
  s_wqm_b64 exec, exec
  s_brev_b32 s0, s1
  s_bcnt1_i32_b32 s0, s1
  s_bcnt1_i32_b64 s0, s[2:3]
  s_bcnt1_i32_b64 s0, exec
  s_ff1_i32_b32 s0, s1
  s_ff1_i32_b64 s0, s[2:3]
  s_flbit_i32_b32 s0, s1
  s_flbit_i32_b64 s0, s[2:3]
  s_flbit_i32 s0, s1
  s_sext_i32_i8 s0, s1
  s_sext_i32_i16 s0, s1
  s_bitset0_b32 s0, s1
  s_bitset1_b32 s0, s1
  s_abs_i32 s0, s1
  s_getpc_b64 s[0:1]
  s_setpc_b64 s[0:1]
  s_swappc_b64 s[30:31], s[0:1]
  s_and_saveexec_b64 s[0:1], vcc
  s_and_saveexec_b64 s[0:1], s[2:3]
  s_or_saveexec_b64 s[0:1], -1
  s_or_saveexec_b64 s[0:1], s[2:3]
  s_xor_saveexec_b64 s[0:1], s[2:3]
  s_andn2_saveexec_b64 s[0:1], s[2:3]
  s_orn2_saveexec_b64 s[0:1], s[2:3]
  s_andn1_saveexec_b64 s[0:1], s[2:3]
  s_movrels_b32 s0, s1
  s_movreld_b32 s0, s1

; ===================================================================================================================
; SOP2: two scalar sources
; ===================================================================================================================
  s_add_u32 s0, s1, s2
  s_add_u32 s0, s1, 4
  s_add_u32 s0, s1, 0x1000
  s_addc_u32 s1, s3, 0
  s_addc_u32 s1, s3, s5
  s_sub_u32 s0, s1, s2
  s_subb_u32 s1, s3, 0
  s_add_i32 s0, s1, -1
  s_add_i32 s0, s1, s2
  s_sub_i32 s0, s1, s2
  s_mul_i32 s0, s1, s2
  s_mul_i32 s0, s1, 0x100
  s_mul_hi_u32 s0, s1, s2
  s_mul_hi_i32 s0, s1, s2
  s_min_u32 s0, s1, s2
  s_min_i32 s0, s1, s2
  s_max_u32 s0, s1, s2
  s_max_i32 s0, s1, 0
  s_lshl_b32 s0, s1, 2
  s_lshl_b32 s0, s1, s2
  s_lshl_b64 s[0:1], s[2:3], 2
  s_lshl_b64 s[0:1], s[2:3], s4
  s_lshr_b32 s0, s1, 5
  s_lshr_b64 s[0:1], s[2:3], 32
  s_ashr_i32 s0, s1, 31
  s_ashr_i64 s[0:1], s[2:3], s4
  s_and_b32 s0, s1, 63
  s_and_b32 s0, s1, 0xffff
  s_and_b64 s[0:1], s[2:3], s[4:5]
  s_and_b64 vcc, exec, vcc
  s_and_b64 s[0:1], exec, -1
  s_or_b32 s0, s1, s2
  s_or_b64 s[0:1], s[2:3], s[4:5]
  s_xor_b32 s0, s1, s2
  s_xor_b64 s[0:1], s[2:3], -1
  s_andn2_b32 s0, s1, s2
  s_andn2_b64 s[0:1], s[2:3], s[4:5]
  s_orn2_b64 s[0:1], s[2:3], s[4:5]
  s_nand_b64 s[0:1], s[2:3], s[4:5]
  s_nor_b64 s[0:1], s[2:3], s[4:5]
  s_xnor_b64 s[0:1], s[2:3], s[4:5]
  s_cselect_b32 s0, s1, s2
  s_cselect_b32 s0, 1.0, 0
  s_cselect_b64 s[0:1], s[2:3], 0
  s_cselect_b64 s[0:1], -1, 0
  s_bfe_u32 s0, s1, 0x80008
  s_bfe_i32 s0, s1, s2
  s_bfe_u64 s[0:1], s[2:3], 0x200000
  s_bfm_b32 s0, s1, s2
  s_bfm_b64 s[0:1], s2, s3
  s_absdiff_i32 s0, s1, s2
  s_lshl1_add_u32 s0, s1, s2
  s_lshl2_add_u32 s0, s1, s2
  s_lshl3_add_u32 s0, s1, s2
  s_lshl4_add_u32 s0, s1, s2
  s_pack_ll_b32_b16 s0, s1, s2
  s_pack_lh_b32_b16 s0, s1, s2
  s_pack_hh_b32_b16 s0, s1, s2

; ===================================================================================================================
; SOPK: a 16-bit constant
; ===================================================================================================================
  s_movk_i32 s0, 0x1234
  s_movk_i32 s0, -1
  s_movk_i32 m0, 0x400
  s_cmovk_i32 s0, 0x10
  s_addk_i32 s0, 0x100
  s_addk_i32 s0, -4
  s_mulk_i32 s0, 3
  s_cmpk_eq_u32 s0, 0x40
  s_cmpk_lg_u32 s0, 0
  s_cmpk_gt_u32 s0, 0x7fff
  s_cmpk_lt_i32 s0, -2
  s_cmpk_ge_i32 s0, 1
  s_getreg_b32 s0, hwreg(HW_REG_HW_ID)
  s_getreg_b32 s0, hwreg(HW_REG_MODE)
  s_getreg_b32 s0, hwreg(HW_REG_MODE, 0, 4)
  s_getreg_b32 s0, hwreg(HW_REG_IB_STS)
  s_setreg_b32 hwreg(HW_REG_MODE, 0, 4), s0
  s_setreg_b32 hwreg(HW_REG_MODE, 4, 4), s1
  s_setreg_imm32_b32 hwreg(HW_REG_MODE, 0, 8), 0xf0
  s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 4), 0x3

; ===================================================================================================================
; SOPC: scalar compares (they set SCC)
; ===================================================================================================================
  s_cmp_eq_u32 s0, s1
  s_cmp_eq_u32 s0, 0
  s_cmp_lg_u32 s0, 0
  s_cmp_gt_u32 s0, s1
  s_cmp_ge_u32 s0, 64
  s_cmp_lt_u32 s0, 0x1000
  s_cmp_le_u32 s0, s1
  s_cmp_eq_i32 s0, -1
  s_cmp_lg_i32 s0, s1
  s_cmp_gt_i32 s0, 0
  s_cmp_ge_i32 s0, s1
  s_cmp_lt_i32 s0, s1
  s_cmp_le_i32 s0, 1
  s_cmp_eq_u64 s[0:1], s[2:3]
  s_cmp_eq_u64 s[0:1], 0
  s_cmp_lg_u64 s[0:1], 0
  s_cmp_lg_u64 exec, 0
  s_bitcmp0_b32 s0, 3
  s_bitcmp1_b32 s0, 31
  s_bitcmp1_b64 s[0:1], 63
  s_set_gpr_idx_on s0, gpr_idx(SRC0)
  s_set_gpr_idx_off

; ===================================================================================================================
; SOPP: program control, waits, barriers and branches to labels
; ===================================================================================================================
loop_top:
  s_nop 0
  s_nop 1
  s_nop 4
  s_nop 7
  s_waitcnt 0
  s_waitcnt vmcnt(0)
  s_waitcnt vmcnt(1)
  s_waitcnt vmcnt(15)
  s_waitcnt vmcnt(16)
  s_waitcnt vmcnt(63)
  s_waitcnt lgkmcnt(0)
  s_waitcnt lgkmcnt(1)
  s_waitcnt lgkmcnt(15)
  s_waitcnt expcnt(0)
  s_waitcnt vmcnt(0) lgkmcnt(0)
  s_waitcnt vmcnt(2) lgkmcnt(1)
  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
  s_waitcnt vmcnt(0) & lgkmcnt(0)
  s_barrier
  s_waitcnt lgkmcnt(0)
  s_barrier
  s_sleep 1
  s_setprio 0
  s_setprio 3
  s_sethalt 0
  s_icache_inv
  s_trap 2
  s_branch loop_top
  s_branch loop_end
  s_cbranch_scc0 loop_top
  s_cbranch_scc1 loop_end
  s_cbranch_vccz loop_top
  s_cbranch_vccnz loop_end
  s_cbranch_execz loop_end
  s_cbranch_execnz loop_top
  s_branch 0
  s_branch -1
  s_cbranch_scc0 3
loop_end:
  s_endpgm

; ===================================================================================================================
; SMEM: scalar loads (kernel arguments, constants) of 1, 2, 4, 8 and 16 dwords
; ===================================================================================================================
  s_load_dword s0, s[4:5], 0x0
  s_load_dword s0, s[4:5], 0x8
  s_load_dword s7, s[4:5], 0xfffff
  s_load_dword s0, s[4:5], s6
  s_load_dword s0, s[4:5], m0
  s_load_dword s0, s[4:5], 0x10 glc
  s_load_dwordx2 s[0:1], s[4:5], 0x0
  s_load_dwordx2 s[2:3], s[4:5], 0x18
  s_load_dwordx2 s[0:1], s[4:5], s6
  s_load_dwordx4 s[0:3], s[4:5], 0x0
  s_load_dwordx4 s[8:11], s[4:5], 0x20
  s_load_dwordx8 s[8:15], s[4:5], 0x0
  s_load_dwordx8 s[16:23], s[4:5], 0x40
  s_load_dwordx16 s[16:31], s[4:5], 0x0
  s_load_dwordx16 s[32:47], s[4:5], 0x80
  s_buffer_load_dword s0, s[8:11], 0x4
  s_buffer_load_dwordx2 s[0:1], s[8:11], 0x8
  s_buffer_load_dwordx4 s[0:3], s[8:11], s4
  s_store_dword s0, s[4:5], 0x0
  s_store_dwordx2 s[0:1], s[4:5], 0x8
  s_dcache_wb
  s_dcache_inv
  s_memtime s[0:1]
  s_memrealtime s[0:1]

; ===================================================================================================================
; VOP1: one vector source (moves, conversions, transcendentals, bit operations)
; ===================================================================================================================
  v_mov_b32 v0, v1
  v_mov_b32 v0, s1
  v_mov_b32 v0, 0
  v_mov_b32 v0, -1
  v_mov_b32 v0, 64
  v_mov_b32 v0, -16
  v_mov_b32 v0, 1.0
  v_mov_b32 v0, -2.0
  v_mov_b32 v0, 0.5
  v_mov_b32 v0, 4.0
  v_mov_b32 v0, 0.15915494
  v_mov_b32 v0, 0x3fb8aa3b
  v_mov_b32 v0, 0x7fffffff
  v_mov_b32 v0, 0x80000000
  v_mov_b32 v0, 0xff800000
  v_mov_b32 v255, v0
  v_mov_b32 v0, vcc_lo
  v_mov_b32 v0, exec_lo
  v_mov_b32 v0, m0
  v_mov_b32_e64 v0, v1
  v_nop
  v_readfirstlane_b32 s0, v1
  v_readfirstlane_b32 s0, v255
  v_cvt_f32_i32 v0, v1
  v_cvt_f32_u32 v0, v1
  v_cvt_u32_f32 v0, v1
  v_cvt_i32_f32 v0, v1
  v_cvt_f16_f32 v0, v1
  v_cvt_f32_f16 v0, v1
  v_cvt_rpi_i32_f32 v0, v1
  v_cvt_flr_i32_f32 v0, v1
  v_cvt_f32_f64 v0, v[2:3]
  v_cvt_f64_f32 v[0:1], v2
  v_cvt_i32_f64 v0, v[2:3]
  v_cvt_f64_i32 v[0:1], v2
  v_cvt_f64_u32 v[0:1], v2
  v_cvt_u32_f64 v0, v[2:3]
  v_cvt_f32_ubyte0 v0, v1
  v_cvt_f32_ubyte1 v0, v1
  v_cvt_f32_ubyte2 v0, v1
  v_cvt_f32_ubyte3 v0, v1
  v_cvt_f16_u16 v0, v1
  v_cvt_f16_i16 v0, v1
  v_cvt_u16_f16 v0, v1
  v_cvt_i16_f16 v0, v1
  v_cvt_off_f32_i4 v0, v1
  v_cvt_norm_i16_f16 v0, v1
  v_fract_f32 v0, v1
  v_trunc_f32 v0, v1
  v_ceil_f32 v0, v1
  v_rndne_f32 v0, v1
  v_floor_f32 v0, v1
  v_exp_f32 v0, v1
  v_exp_f32 v0, s1
  v_log_f32 v0, v1
  v_rcp_f32 v0, v1
  v_rcp_iflag_f32 v0, v1
  v_rsq_f32 v0, v1
  v_sqrt_f32 v0, v1
  v_sin_f32 v0, v1
  v_cos_f32 v0, v1
  v_exp_f16 v0, v1
  v_log_f16 v0, v1
  v_rcp_f16 v0, v1
  v_rsq_f16 v0, v1
  v_sqrt_f16 v0, v1
  v_rcp_f64 v[0:1], v[2:3]
  v_rsq_f64 v[0:1], v[2:3]
  v_sqrt_f64 v[0:1], v[2:3]
  v_frexp_exp_i32_f32 v0, v1
  v_frexp_mant_f32 v0, v1
  v_frexp_exp_i32_f64 v0, v[2:3]
  v_not_b32 v0, v1
  v_bfrev_b32 v0, v1
  v_ffbh_u32 v0, v1
  v_ffbl_b32 v0, v1
  v_ffbh_i32 v0, v1
  v_mov_b64 v[0:1], v[2:3]
  v_mov_b64 v[0:1], s[2:3]
  v_swap_b32 v0, v1

; ===================================================================================================================
; VOP2: two vector sources (src0 may be a scalar, an inline constant or a 32-bit literal)
; ===================================================================================================================
  v_add_f32 v0, v1, v2
  v_add_f32 v0, s1, v2
  v_add_f32 v0, 1.0, v2
  v_add_f32 v0, -4.0, v2
  v_add_f32 v0, 0x3f000001, v2
  v_add_f32 v0, 0.15915494, v2
  v_sub_f32 v0, v1, v2
  v_subrev_f32 v0, v1, v2
  v_mul_f32 v0, v1, v2
  v_mul_f32 v0, 0x3fb8aa3b, v1
  v_mul_f32 v0, s1, v2
  v_mul_legacy_f32 v0, v1, v2
  v_mac_f32 v0, v1, v2
  v_mac_f32 v0, s1, v2
  v_fmac_f32 v0, v1, v2
  v_fmac_f32 v0, s1, v2
  v_fmac_f32 v0, 0.5, v2
  v_madmk_f32 v0, v1, 0x3fb8aa3b, v2
  v_madak_f32 v0, v1, v2, 0x3f800000
  v_max_f32 v0, v1, v2
  v_max_f32 v0, 0, v2
  v_min_f32 v0, v1, v2
  v_max_i32 v0, v1, v2
  v_min_i32 v0, v1, v2
  v_max_u32 v0, v1, v2
  v_min_u32 v0, v1, v2
  v_mul_i32_i24 v0, v1, v2
  v_mul_u32_u24 v0, v1, v2
  v_mul_u32_u24 v0, s1, v2
  v_mul_hi_u32_u24 v0, v1, v2
  v_mul_hi_i32_i24 v0, v1, v2
  v_lshrrev_b32 v0, 2, v1
  v_lshrrev_b32 v0, v1, v2
  v_ashrrev_i32 v0, 31, v1
  v_lshlrev_b32 v0, 2, v1
  v_lshlrev_b32 v0, 4, v1
  v_lshlrev_b32 v0, s1, v2
  v_lshlrev_b16 v0, 8, v1
  v_and_b32 v0, v1, v2
  v_and_b32 v0, 63, v1
  v_and_b32 v0, 0xffff, v1
  v_and_b32 v0, 0x7fffffff, v1
  v_or_b32 v0, v1, v2
  v_xor_b32 v0, v1, v2
  v_xor_b32 v0, 0x80000000, v1
  v_add_u32 v0, v1, v2
  v_add_u32 v0, s1, v2
  v_add_u32 v0, 16, v2
  v_sub_u32 v0, v1, v2
  v_subrev_u32 v0, v1, v2
  v_add_co_u32 v0, vcc, v1, v2
  v_add_co_u32 v0, vcc, s1, v2
  v_addc_co_u32 v1, vcc, v3, v4, vcc
  v_addc_co_u32 v1, vcc, 0, v4, vcc
  v_sub_co_u32 v0, vcc, v1, v2
  v_subb_co_u32 v1, vcc, v3, v4, vcc
  v_subrev_co_u32 v0, vcc, v1, v2
  v_cndmask_b32 v0, v1, v2, vcc
  v_cndmask_b32 v0, 0, v2, vcc
  v_cndmask_b32 v0, 1.0, v2, vcc
  v_add_f16 v0, v1, v2
  v_sub_f16 v0, v1, v2
  v_mul_f16 v0, v1, v2
  v_mac_f16 v0, v1, v2
  v_max_f16 v0, v1, v2
  v_min_f16 v0, v1, v2
  v_ldexp_f16 v0, v1, v2
  v_add_u16 v0, v1, v2
  v_mul_lo_u16 v0, v1, v2
  v_ldexp_f32 v0, v1, v2
  v_fmac_f64 v[0:1], v[2:3], v[4:5]
  v_dot2c_f32_f16 v0, v1, v2
  v_dot4c_i32_i8 v0, v1, v2
  v_xnor_b32 v0, v1, v2

; ===================================================================================================================
; VOPC: vector compares, into VCC (e32) or any SGPR pair (e64); v_cmpx writes EXEC
; ===================================================================================================================
  v_cmp_lt_f32 vcc, v1, v2
  v_cmp_eq_f32 vcc, v1, v2
  v_cmp_le_f32 vcc, v1, v2
  v_cmp_gt_f32 vcc, v1, v2
  v_cmp_lg_f32 vcc, v1, v2
  v_cmp_ge_f32 vcc, v1, v2
  v_cmp_o_f32 vcc, v1, v2
  v_cmp_u_f32 vcc, v1, v1
  v_cmp_neq_f32 vcc, v1, v2
  v_cmp_nlt_f32 vcc, v1, v2
  v_cmp_lt_f32 vcc, 0, v2
  v_cmp_gt_f32 vcc, s1, v2
  v_cmp_lt_f32 vcc, 0x3f000000, v2
  v_cmp_eq_u32 vcc, v1, v2
  v_cmp_ne_u32 vcc, 0, v2
  v_cmp_lt_u32 vcc, v1, v2
  v_cmp_gt_u32 vcc, s1, v2
  v_cmp_le_u32 vcc, v1, v2
  v_cmp_ge_u32 vcc, v1, v2
  v_cmp_lt_i32 vcc, v1, v2
  v_cmp_gt_i32 vcc, 0, v2
  v_cmp_eq_i32 vcc, v1, v2
  v_cmp_ne_i32 vcc, v1, v2
  v_cmp_eq_u64 vcc, v[0:1], v[2:3]
  v_cmp_lt_u64 vcc, v[0:1], v[2:3]
  v_cmp_lt_f64 vcc, v[0:1], v[2:3]
  v_cmp_lt_f16 vcc, v1, v2
  v_cmp_class_f32 vcc, v1, v2
  v_cmp_class_f32 vcc, v1, 3
  v_cmp_class_f32_e64 s[0:1], v1, s2
  v_cmp_lt_f32_e64 s[0:1], v1, v2
  v_cmp_lt_f32_e64 s[0:1], v1, s2
  v_cmp_lt_f32_e64 s[0:1], v1, 1.0
  v_cmp_lt_f32_e64 s[0:1], |v1|, v2
  v_cmp_lt_f32_e64 s[0:1], -v1, v2
  v_cmp_gt_u32_e64 s[0:1], s2, v1
  v_cmp_eq_u32_e64 s[2:3], v1, 0
  v_cmp_ne_u32_e64 vcc, v1, s2
  v_cmp_eq_u32_e64 s[0:1], s2, s2
  v_cmpx_lt_u32 vcc, v1, v2
  v_cmpx_gt_u32 vcc, s1, v2
  v_cmpx_eq_u32 vcc, 0, v2
  v_cmpx_lt_f32 vcc, v1, v2
  v_cmpx_lt_u32_e64 s[0:1], v1, v2

; ===================================================================================================================
; VOP3: three sources, any SGPR or VCC carry, input modifiers (neg, abs), clamp and output modifiers (omod)
; ===================================================================================================================
  v_add_f32_e64 v0, v1, v2
  v_add_f32_e64 v0, -v1, v2
  v_add_f32_e64 v0, |v1|, v2
  v_add_f32_e64 v0, -|v1|, v2
  v_add_f32_e64 v0, v1, -v2
  v_add_f32_e64 v0, v1, v2 clamp
  v_add_f32_e64 v0, v1, v2 mul:2
  v_add_f32_e64 v0, v1, v2 mul:4
  v_add_f32_e64 v0, v1, v2 div:2
  v_add_f32_e64 v0, v1, v2 clamp mul:2
  v_add_f32_e64 v0, s1, s1
  v_add_f32_e64 v0, v1, 1.0
  v_add_f32_e64 v0, v1, -0.5
  v_mul_f32_e64 v0, v1, s2
  v_mul_f32_e64 v0, -v1, v2
  v_max_f32_e64 v0, |v1|, |v2|
  v_max_f32_e64 v0, v1, v1 clamp
  v_exp_f32_e64 v0, -v1
  v_log_f32_e64 v0, |v1|
  v_rcp_f32_e64 v0, v1 clamp
  v_rsq_f32_e64 v0, |v1|
  v_sqrt_f32_e64 v0, |v1|
  v_sin_f32_e64 v0, v1 mul:2
  v_cvt_f32_i32_e64 v0, v1 div:2
  v_fma_f32 v0, v1, v2, v3
  v_fma_f32 v0, v1, s2, v3
  v_fma_f32 v0, v1, v2, 1.0
  v_fma_f32 v0, -v1, v2, v3
  v_fma_f32 v0, v1, v2, -v3
  v_fma_f32 v0, |v1|, v2, v3
  v_fma_f32 v0, v1, v2, v3 clamp
  v_fma_f32 v0, v1, 0.5, 0.5
  v_mad_f32 v0, v1, v2, v3
  v_mad_legacy_f32 v0, v1, v2, v3
  v_fma_f64 v[0:1], v[2:3], v[4:5], v[6:7]
  v_fma_f16 v0, v1, v2, v3
  v_mad_f16 v0, v1, v2, v3
  v_add_f64 v[0:1], v[2:3], v[4:5]
  v_add_f64 v[0:1], -v[2:3], v[4:5]
  v_mul_f64 v[0:1], v[2:3], v[4:5]
  v_max_f64 v[0:1], v[2:3], v[4:5]
  v_ldexp_f64 v[0:1], v[2:3], v4
  v_ldexp_f32 v0, v1, s2
  v_mad_u32_u24 v0, v1, v2, v3
  v_mad_i32_i24 v0, v1, v2, v3
  v_mad_u64_u32 v[0:1], s[2:3], v4, v5, v[6:7]
  v_mad_u64_u32 v[0:1], s[2:3], v4, s5, 0
  v_mad_i64_i32 v[0:1], s[2:3], v4, v5, v[6:7]
  v_mul_lo_u32 v0, v1, v2
  v_mul_lo_u32 v0, s1, v2
  v_mul_hi_u32 v0, v1, v2
  v_mul_hi_i32 v0, v1, v2
  v_add_co_u32_e64 v0, s[0:1], v1, v2
  v_addc_co_u32_e64 v1, s[0:1], v3, v4, s[0:1]
  v_sub_co_u32_e64 v0, s[0:1], v1, v2
  v_add_u32_e64 v0, v1, v2 clamp
  v_add3_u32 v0, v1, v2, v3
  v_add3_u32 v0, v1, s2, 4
  v_lshl_add_u32 v0, v1, 2, v2
  v_lshl_add_u32 v0, v1, 2, s2
  v_add_lshl_u32 v0, v1, v2, 2
  v_lshl_or_b32 v0, v1, 16, v2
  v_and_or_b32 v0, v1, s3, v2
  v_or3_b32 v0, v1, v2, v3
  v_xad_u32 v0, v1, v2, v3
  v_lshlrev_b64 v[0:1], 2, v[2:3]
  v_lshrrev_b64 v[0:1], v2, v[4:5]
  v_ashrrev_i64 v[0:1], 31, v[2:3]
  v_bfe_u32 v0, v1, 8, 8
  v_bfe_u32 v0, v1, v2, v3
  v_bfe_i32 v0, v1, 0, 16
  v_bfi_b32 v0, v1, v2, v3
  v_bfm_b32 v0, v1, v2
  v_alignbit_b32 v0, v1, v2, 16
  v_alignbyte_b32 v0, v1, v2, 1
  v_perm_b32 v0, v1, v2, v3
  v_perm_b32 v0, v1, v2, s3
  v_bcnt_u32_b32 v0, v1, 0
  v_bcnt_u32_b32 v0, v1, v2
  v_mbcnt_lo_u32_b32 v0, -1, 0
  v_mbcnt_hi_u32_b32 v0, -1, v0
  v_cndmask_b32_e64 v0, v1, v2, s[0:1]
  v_cndmask_b32_e64 v0, 0, 1.0, vcc
  v_cndmask_b32_e64 v0, -v1, v2, s[0:1]
  v_med3_f32 v0, v1, v2, v3
  v_med3_f32 v0, v1, 0, 1.0
  v_med3_i32 v0, v1, v2, v3
  v_med3_u32 v0, v1, v2, v3
  v_max3_f32 v0, v1, v2, v3
  v_min3_f32 v0, v1, v2, v3
  v_max3_u32 v0, v1, v2, v3
  v_min3_i32 v0, v1, v2, v3
  v_sad_u32 v0, v1, v2, v3
  v_cvt_pkrtz_f16_f32 v0, v1, v2
  v_cvt_pknorm_i16_f32 v0, v1, v2
  v_cvt_pk_u16_u32 v0, v1, v2
  v_cvt_pk_i16_i32 v0, v1, v2
  v_cvt_pk_u8_f32 v0, v1, 0, v2
  v_pack_b32_f16 v0, v1, v2
  v_pack_b32_f16 v0, v1, v2 op_sel:[1,1,0]
  v_div_scale_f32 v0, vcc, v1, v2, v1
  v_div_scale_f32 v0, vcc, v2, v2, v1
  v_div_scale_f32 v0, s[2:3], v1, v2, v1
  v_div_fmas_f32 v0, v1, v2, v3
  v_div_fixup_f32 v0, v1, v2, v3
  v_div_scale_f64 v[0:1], vcc, v[2:3], v[4:5], v[2:3]
  v_div_fmas_f64 v[0:1], v[2:3], v[4:5], v[6:7]
  v_div_fixup_f64 v[0:1], v[2:3], v[4:5], v[6:7]
  v_div_fixup_f16 v0, v1, v2, v3
  v_trig_preop_f64 v[0:1], v[2:3], v4
  v_readlane_b32 s0, v1, 5
  v_readlane_b32 s0, v1, 63
  v_readlane_b32 s0, v1, s2
  v_readlane_b32 s0, v1, m0
  v_writelane_b32 v0, s1, 0
  v_writelane_b32 v0, s1, 63
  v_writelane_b32 v0, s1, 7
  v_writelane_b32 v0, 0, s2
  v_mad_u16 v0, v1, v2, v3
  v_mad_mixlo_f16 v0, v1, v2, v3

; ===================================================================================================================
; VOP3P: packed 16-bit pairs and, on CDNA 2 and later, packed float32; mixed-precision and dot products
; ===================================================================================================================
  v_pk_add_f16 v0, v1, v2
  v_pk_add_f16 v0, v1, v2 op_sel:[1,0]
  v_pk_add_f16 v0, v1, v2 op_sel_hi:[0,1]
  v_pk_add_f16 v0, v1, v2 neg_lo:[1,0] neg_hi:[1,0]
  v_pk_add_f16 v0, v1, v2 clamp
  v_pk_mul_f16 v0, v1, v2
  v_pk_mul_f16 v0, v1, s2
  v_pk_mul_f16 v0, v1, 1.0
  v_pk_fma_f16 v0, v1, v2, v3
  v_pk_fma_f16 v0, v1, v2, v3 op_sel:[0,1,0] op_sel_hi:[0,1,1]
  v_pk_fma_f16 v0, v1, v2, v3 neg_lo:[0,0,1] neg_hi:[0,0,1]
  v_pk_max_f16 v0, v1, v2
  v_pk_min_f16 v0, v1, v2
  v_pk_add_u16 v0, v1, v2
  v_pk_add_i16 v0, v1, v2
  v_pk_sub_i16 v0, v1, v2
  v_pk_mul_lo_u16 v0, v1, v2
  v_pk_mad_u16 v0, v1, v2, v3
  v_pk_lshlrev_b16 v0, 1, v2
  v_pk_ashrrev_i16 v0, v1, v2
  v_mad_mix_f32 v0, v1, v2, v3
  v_mad_mix_f32 v0, v1, v2, v3 op_sel:[1,0,0] op_sel_hi:[1,1,0]
  v_fma_mix_f32 v0, v1, v2, v3
  v_fma_mix_f32 v0, v1, v2, v3 op_sel_hi:[1,1,0]
  v_fma_mixlo_f16 v0, v1, v2, v3
  v_fma_mixhi_f16 v0, v1, v2, v3
  v_dot2_f32_f16 v0, v1, v2, v3
  v_dot2_i32_i16 v0, v1, v2, v3
  v_dot2_u32_u16 v0, v1, v2, v3
  v_dot4_i32_i8 v0, v1, v2, v3
  v_dot4_u32_u8 v0, v1, v2, v3
  v_dot8_i32_i4 v0, v1, v2, v3
  v_dot8_u32_u4 v0, v1, v2, v3
  v_pk_fma_f32 v[0:1], v[2:3], v[4:5], v[6:7]
  v_pk_fma_f32 v[0:1], v[2:3], s[4:5], v[6:7] op_sel_hi:[1,0,1]
  v_pk_mul_f32 v[0:1], v[2:3], v[4:5]
  v_pk_add_f32 v[0:1], v[2:3], v[4:5]
  v_pk_add_f32 v[0:1], v[2:3], v[4:5] neg_lo:[0,1] neg_hi:[0,1]
  v_pk_mov_b32 v[0:1], v[2:3], v[4:5] op_sel:[0,1]

; ===================================================================================================================
; DS: the workgroup's shared memory (LDS) and the wave's lane exchanges
; ===================================================================================================================
  ds_read_b32 v0, v1
  ds_read_b32 v0, v1 offset:4
  ds_read_b32 v0, v1 offset:65535
  ds_read_b64 v[0:1], v2
  ds_read_b64 v[0:1], v2 offset:256
  ds_read_b128 v[0:3], v4
  ds_read_b128 v[0:3], v4 offset:512
  ds_read2_b32 v[0:1], v2 offset1:1
  ds_read2_b32 v[0:1], v2 offset0:4 offset1:8
  ds_read2st64_b32 v[0:1], v2 offset1:1
  ds_read2_b64 v[0:3], v4 offset1:1
  ds_read_u8 v0, v1
  ds_read_i8 v0, v1
  ds_read_u16 v0, v1 offset:2
  ds_read_i16 v0, v1
  ds_read_u16_d16 v0, v1
  ds_read_u16_d16_hi v0, v1
  ds_write_b32 v1, v0
  ds_write_b32 v1, v0 offset:4
  ds_write_b64 v2, v[0:1]
  ds_write_b64 v2, v[0:1] offset:8
  ds_write_b128 v4, v[0:3]
  ds_write_b128 v4, v[0:3] offset:16
  ds_write2_b32 v2, v0, v1 offset1:1
  ds_write2_b32 v2, v0, v1 offset0:2 offset1:3
  ds_write2st64_b32 v2, v0, v1 offset1:1
  ds_write2_b64 v4, v[0:1], v[2:3] offset1:1
  ds_write_b8 v1, v0
  ds_write_b16 v1, v0 offset:2
  ds_write_b8_d16_hi v1, v0
  ds_write_b16_d16_hi v1, v0
  ds_add_u32 v1, v0
  ds_add_u32 v1, v0 offset:4
  ds_add_rtn_u32 v0, v1, v2
  ds_add_f32 v1, v0
  ds_add_rtn_f32 v0, v1, v2
  ds_sub_u32 v1, v0
  ds_max_f32 v1, v0
  ds_min_f32 v1, v0
  ds_max_u32 v1, v0
  ds_add_u64 v2, v[0:1]
  ds_add_f64 v2, v[0:1]
  ds_cmpst_rtn_b32 v0, v1, v2, v3
  ds_wrxchg_rtn_b32 v0, v1, v2
  ds_bpermute_b32 v0, v1, v2
  ds_bpermute_b32 v0, v1, v2 offset:4
  ds_permute_b32 v0, v1, v2
  ds_swizzle_b32 v0, v1 offset:swizzle(SWAP,1)
  ds_swizzle_b32 v0, v1 offset:swizzle(SWAP,16)
  ds_swizzle_b32 v0, v1 offset:swizzle(REVERSE,8)
  ds_swizzle_b32 v0, v1 offset:swizzle(BROADCAST,4,0)
  ds_swizzle_b32 v0, v1 offset:swizzle(QUAD_PERM,1,0,3,2)
  ds_swizzle_b32 v0, v1 offset:swizzle(BITMASK_PERM,"01pip")
  ds_swizzle_b32 v0, v1 offset:0x041f
  ds_append v0
  ds_consume v0

; ===================================================================================================================
; GLOBAL: memory through a 64-bit VGPR address ("off") or an SGPR base with a 32-bit VGPR offset; signed 13-bit
; offsets; bytes, shorts and 1, 2, 3 and 4 dwords; atomics
; ===================================================================================================================
  global_load_dword v0, v[2:3], off
  global_load_dword v0, v[2:3], off offset:16
  global_load_dword v0, v[2:3], off offset:-16
  global_load_dword v0, v[2:3], off offset:4095
  global_load_dword v0, v[2:3], off offset:-4096
  global_load_dword v0, v1, s[2:3]
  global_load_dword v0, v1, s[2:3] offset:64
  global_load_dword v0, v1, s[2:3] offset:-64
  global_load_dword v0, v1, s[40:41] offset:4
  global_load_dword v0, v[2:3], off glc
  global_load_dword v0, v[2:3], off slc
  global_load_dword v0, v[2:3], off glc slc
  global_load_dword v0, v[2:3], off sc0
  global_load_dword v0, v[2:3], off sc1
  global_load_dword v0, v[2:3], off nt
  global_load_dword v0, v[2:3], off sc0 sc1 nt
  global_load_dwordx2 v[0:1], v[2:3], off
  global_load_dwordx2 v[0:1], v2, s[4:5] offset:8
  global_load_dwordx3 v[0:2], v[4:5], off
  global_load_dwordx4 v[0:3], v[4:5], off
  global_load_dwordx4 v[0:3], v4, s[6:7] offset:16
  global_load_dwordx4 v[4:7], v0, s[2:3]
  global_load_ubyte v0, v[2:3], off
  global_load_ubyte v0, v1, s[2:3] offset:1
  global_load_sbyte v0, v[2:3], off
  global_load_ushort v0, v[2:3], off
  global_load_ushort v0, v1, s[2:3] offset:2
  global_load_sshort v0, v[2:3], off
  global_load_short_d16 v0, v[2:3], off
  global_load_short_d16_hi v0, v[2:3], off
  global_load_ubyte_d16 v0, v[2:3], off
  global_load_ubyte_d16_hi v0, v[2:3], off
  global_store_dword v[2:3], v0, off
  global_store_dword v[2:3], v0, off offset:-4
  global_store_dword v1, v0, s[2:3]
  global_store_dword v1, v0, s[2:3] offset:4
  global_store_dword v[2:3], v0, off glc
  global_store_dword v[2:3], v0, off sc0 sc1
  global_store_dwordx2 v[2:3], v[0:1], off
  global_store_dwordx2 v4, v[0:1], s[2:3] offset:8
  global_store_dwordx3 v[4:5], v[0:2], off
  global_store_dwordx4 v[4:5], v[0:3], off
  global_store_dwordx4 v4, v[0:3], s[2:3] offset:16
  global_store_byte v[2:3], v0, off
  global_store_byte v1, v0, s[2:3] offset:3
  global_store_byte_d16_hi v[2:3], v0, off
  global_store_short v[2:3], v0, off
  global_store_short v1, v0, s[2:3] offset:6
  global_store_short_d16_hi v[2:3], v0, off
  global_atomic_add v[2:3], v0, off
  global_atomic_add v1, v0, s[2:3]
  global_atomic_add v0, v[2:3], v1, off glc
  global_atomic_add v0, v[2:3], v1, off sc0
  global_atomic_umax v[2:3], v0, off
  global_atomic_cmpswap v0, v[2:3], v[4:5], off glc
  global_atomic_add_x2 v[2:3], v[4:5], off
  global_atomic_add_f32 v[2:3], v0, off
  global_atomic_add_f32 v1, v0, s[2:3]
  global_atomic_add_f32 v0, v[2:3], v1, off glc
  global_atomic_pk_add_f16 v[2:3], v0, off
  global_atomic_add_f64 v[2:3], v[4:5], off

; ===================================================================================================================
; FLAT: memory through a 64-bit VGPR address in any segment; unsigned 12-bit offsets
; ===================================================================================================================
  flat_load_dword v0, v[2:3]
  flat_load_dword v0, v[2:3] offset:16
  flat_load_dword v0, v[2:3] offset:4095
  flat_load_dword v0, v[2:3] glc
  flat_load_dword v0, v[2:3] sc0 sc1
  flat_load_dwordx2 v[0:1], v[2:3]
  flat_load_dwordx4 v[0:3], v[4:5]
  flat_load_dwordx4 v[0:3], v[4:5] offset:32
  flat_load_ubyte v0, v[2:3]
  flat_load_sbyte v0, v[2:3]
  flat_load_ushort v0, v[2:3]
  flat_load_sshort v0, v[2:3]
  flat_store_dword v[2:3], v0
  flat_store_dword v[2:3], v0 offset:4
  flat_store_dwordx2 v[2:3], v[0:1]
  flat_store_dwordx4 v[4:5], v[0:3]
  flat_store_byte v[2:3], v0
  flat_store_short v[2:3], v0
  flat_atomic_add v[2:3], v0
  flat_atomic_add v0, v[2:3], v1 glc

; ===================================================================================================================
; DPP: a lane's neighbours within a row of 16 or the wave, as VOP1 and VOP2 sources
; ===================================================================================================================
  v_mov_b32_dpp v0, v1 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf
  v_mov_b32_dpp v0, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
  v_mov_b32_dpp v0, v1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
  v_mov_b32_dpp v0, v1 row_shl:1 row_mask:0xf bank_mask:0xf
  v_mov_b32_dpp v0, v1 row_shr:1 row_mask:0xf bank_mask:0xf
  v_mov_b32_dpp v0, v1 row_shr:2 row_mask:0xf bank_mask:0xf
  v_mov_b32_dpp v0, v1 row_shr:4 row_mask:0xf bank_mask:0xe
  v_mov_b32_dpp v0, v1 row_shr:8 row_mask:0xf bank_mask:0xc
  v_mov_b32_dpp v0, v1 row_ror:4 row_mask:0xf bank_mask:0xf
  v_mov_b32_dpp v0, v1 row_ror:8 row_mask:0xf bank_mask:0xf
  v_mov_b32_dpp v0, v1 row_mirror row_mask:0xf bank_mask:0xf
  v_mov_b32_dpp v0, v1 row_half_mirror row_mask:0xf bank_mask:0xf
  v_mov_b32_dpp v0, v1 row_bcast:15 row_mask:0xa bank_mask:0xf
  v_mov_b32_dpp v0, v1 row_bcast:31 row_mask:0xc bank_mask:0xf
  v_mov_b32_dpp v0, v1 wave_shl:1 row_mask:0xf bank_mask:0xf
  v_mov_b32_dpp v0, v1 wave_shr:1 row_mask:0xf bank_mask:0xf
  v_mov_b32_dpp v0, v1 wave_rol:1 row_mask:0xf bank_mask:0xf
  v_mov_b32_dpp v0, v1 wave_ror:1 row_mask:0xf bank_mask:0xf
  v_mov_b32_dpp v0, v1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:0
  v_mov_b32_dpp v0, v1 row_newbcast:1 row_mask:0xf bank_mask:0xf
  v_add_f32_dpp v0, v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
  v_add_f32_dpp v0, v1, v0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:0
  v_add_f32_dpp v0, v1, v0 row_bcast:15 row_mask:0xa bank_mask:0xf
  v_add_f32_dpp v0, -v1, |v2| row_shr:2 row_mask:0xf bank_mask:0xf
  v_max_f32_dpp v0, v1, v0 row_shr:4 row_mask:0xf bank_mask:0xf
  v_min_f32_dpp v0, v1, v0 row_ror:8 row_mask:0xf bank_mask:0xf
  v_add_u32_dpp v0, v1, v0 row_shr:1 row_mask:0xf bank_mask:0xf
  v_max_i32_dpp v0, v1, v0 wave_shr:1 row_mask:0xf bank_mask:0xf
  v_mul_f32_dpp v0, v1, v2 row_mirror row_mask:0xf bank_mask:0xf
  v_fmac_f32_dpp v0, v1, v2 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf
  v_cvt_f32_i32_dpp v0, v1 row_shr:1 row_mask:0xf bank_mask:0xf

; ===================================================================================================================
; SDWA: sub-dword selection of sources and destination
; ===================================================================================================================
  v_mov_b32_sdwa v0, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
  v_mov_b32_sdwa v0, v1 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
  v_mov_b32_sdwa v0, v1 dst_sel:BYTE_1 dst_unused:UNUSED_SEXT src0_sel:BYTE_3
  v_mov_b32_sdwa v0, s1 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:WORD_1
  v_cvt_f32_f16_sdwa v0, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
  v_cvt_f16_f32_sdwa v0, v1 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:DWORD
  v_cvt_f32_ubyte0_sdwa v0, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
  v_add_f32_sdwa v0, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_0
  v_add_f16_sdwa v0, v1, v2 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_1
  v_mul_f16_sdwa v0, -v1, |v2| dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_1
  v_and_b32_sdwa v0, v1, v2 dst_sel:BYTE_0 dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:DWORD
  v_or_b32_sdwa v0, v1, v2 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
  v_lshlrev_b32_sdwa v0, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
  v_add_u32_sdwa v0, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_1
  v_add_f32_sdwa v0, v1, v2 clamp dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
  v_cmp_lt_f16_sdwa vcc, v1, v2 src0_sel:WORD_1 src1_sel:WORD_0
  v_cmp_eq_u32_sdwa s[0:1], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD

; ===================================================================================================================
; EXEC and VCC: a divergent if, else and loop, and the masks they leave
; ===================================================================================================================
  v_cmp_gt_u32 vcc, s2, v0
  s_and_saveexec_b64 s[4:5], vcc
  s_cbranch_execz if_end
  v_mov_b32 v1, 1.0
  s_or_b64 exec, exec, s[4:5]
if_end:
  s_or_b64 exec, exec, s[4:5]
  v_cmp_lt_f32 vcc, 0, v1
  s_and_saveexec_b64 s[6:7], vcc
  s_xor_b64 s[6:7], exec, s[6:7]
  s_cbranch_execz else_part
  v_mov_b32 v2, 1.0
else_part:
  s_andn2_saveexec_b64 s[6:7], s[6:7]
  s_cbranch_execz else_end
  v_mov_b32 v2, 0
else_end:
  s_or_b64 exec, exec, s[6:7]
  s_mov_b64 s[8:9], exec
lane_loop:
  v_add_u32 v3, 1, v3
  v_cmp_ge_u32 vcc, v3, v4
  s_andn2_b64 exec, exec, vcc
  s_cbranch_execnz lane_loop
  s_mov_b64 exec, s[8:9]
  s_mov_b64 exec, -1
  s_mov_b32 exec_lo, -1
  s_mov_b32 exec_hi, 0
  s_mov_b64 vcc, 0
  s_mov_b64 vcc, exec
  s_mov_b64 vcc, s[0:1]
  s_and_b64 exec, exec, s[0:1]
  s_andn2_b64 exec, exec, s[0:1]
  s_xor_b64 exec, exec, -1
  s_cmp_eq_u64 exec, 0
  s_cbranch_scc1 if_end
  s_and_b64 vcc, exec, vcc
  s_cbranch_vccz else_end
  s_cbranch_vccnz lane_loop
  s_cbranch_scc0 lane_loop
  s_cbranch_execnz else_part
  s_bcnt1_i32_b64 s0, exec
  s_ff1_i32_b64 s0, exec

; ===================================================================================================================
; Forms only some gfx9 CPUs take, kept to show where they part
; ===================================================================================================================
  v_cvt_f64_f32 v[1:2], v3        ; an unaligned VGPR pair (gfx90a and later require even pairs)
  global_load_dwordx2 v[1:2], v[4:5], off
  v_accvgpr_read_b32 v0, a1
  v_accvgpr_write_b32 a0, v1
  v_accvgpr_write_b32 a0, 0
  v_accvgpr_mov_b32 a0, a1
  s_endpgm
