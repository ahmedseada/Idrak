; The gfx11 instruction forms Idrak's GCN kernels use (plan 35, G8a), in gfx11's own syntax as llvm-mc prints
; it for gfx1100: made by tools/gcn/make_family_forms.py from gfx9/forms.s and the kernels' mnemonics; rerun it,
; do not edit. Wave64 forms first, each followed by its wave32 spelling where that differs; make_gcn_golden.py
; assembles the file for each CPU and wave size and lists the lines each refuses.
s_mov_b32 s0, s1
s_mov_b32 s0, 0
s_mov_b32 s0, 64
s_mov_b32 s0, -1
s_mov_b32 s0, -16
s_mov_b32 s0, 1.0
s_mov_b32 s0, -0.5
s_mov_b32 s0, 0.15915494
s_mov_b32 s0, 0x41
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
s_not_b64 s[0:1], s[2:3]
s_and_saveexec_b64 s[0:1], vcc
s_and_saveexec_b32 s0, vcc_lo
s_and_saveexec_b64 s[0:1], s[2:3]
s_add_u32 s0, s1, s2
s_add_u32 s0, s1, 4
s_add_u32 s0, s1, 0x1000
s_addc_u32 s1, s3, 0
s_addc_u32 s1, s3, s5
s_sub_u32 s0, s1, s2
s_add_i32 s0, s1, -1
s_add_i32 s0, s1, s2
s_sub_i32 s0, s1, s2
s_mul_i32 s0, s1, s2
s_mul_i32 s0, s1, 0x100
s_mul_hi_u32 s0, s1, s2
s_min_u32 s0, s1, s2
s_min_i32 s0, s1, s2
s_max_u32 s0, s1, s2
s_max_i32 s0, s1, 0
s_lshl_b32 s0, s1, 2
s_lshl_b32 s0, s1, s2
s_lshl_b64 s[0:1], s[2:3], 2
s_lshl_b64 s[0:1], s[2:3], s4
s_lshr_b32 s0, s1, 5
s_ashr_i32 s0, s1, 31
s_and_b32 s0, s1, 63
s_and_b32 s0, s1, 0xffff
s_and_b64 s[0:1], s[2:3], s[4:5]
s_and_b64 vcc, exec, vcc
s_and_b64 s[0:1], exec, -1
s_or_b32 s0, s1, s2
s_or_b64 s[0:1], s[2:3], s[4:5]
s_xor_b32 s0, s1, s2
s_xor_b64 s[0:1], s[2:3], -1
s_and_not1_b64 s[0:1], s[2:3], s[4:5]
s_cselect_b32 s0, s1, s2
s_cselect_b32 s0, 1.0, 0
s_cselect_b64 s[0:1], s[2:3], 0
s_cselect_b64 s[0:1], -1, 0
s_movk_i32 s0, 0x1234
s_movk_i32 s0, 0xffff
s_movk_i32 m0, 0x400
s_cmp_eq_u32 s0, s1
s_cmp_eq_u32 s0, 0
s_cmp_lg_u32 s0, 0
s_cmp_gt_u32 s0, s1
s_cmp_ge_u32 s0, 64
s_cmp_lt_u32 s0, 0x1000
s_cmp_le_u32 s0, s1
s_cmp_gt_i32 s0, 0
s_cmp_lt_i32 s0, s1
s_cmp_eq_u64 s[0:1], s[2:3]
s_cmp_eq_u64 s[0:1], 0
s_bitcmp1_b32 s0, 31
loop_top:
s_nop 0
s_nop 1
s_nop 4
s_nop 7
s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
s_waitcnt vmcnt(0)
s_waitcnt vmcnt(1)
s_waitcnt vmcnt(15)
s_waitcnt vmcnt(16)
s_waitcnt vmcnt(63) expcnt(7) lgkmcnt(63)
s_waitcnt lgkmcnt(0)
s_waitcnt lgkmcnt(1)
s_waitcnt lgkmcnt(15)
s_waitcnt expcnt(0)
s_waitcnt vmcnt(0) lgkmcnt(0)
s_waitcnt vmcnt(2) lgkmcnt(1)
s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
s_waitcnt vmcnt(0) lgkmcnt(0)
s_barrier
s_waitcnt lgkmcnt(0)
s_barrier
s_sleep 1
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
s_load_b32 s0, s[4:5], 0x0
s_load_b32 s0, s[4:5], 0x8
s_load_b32 s7, s[4:5], 0xfffff
s_load_b32 s0, s[4:5], s6
s_load_b32 s0, s[4:5], m0
s_load_b32 s0, s[4:5], 0x10 glc
s_load_b64 s[0:1], s[4:5], 0x0
s_load_b64 s[2:3], s[4:5], 0x18
s_load_b64 s[0:1], s[4:5], s6
s_load_b128 s[0:3], s[4:5], 0x0
s_load_b128 s[8:11], s[4:5], 0x20
s_load_b256 s[8:15], s[4:5], 0x0
s_load_b256 s[16:23], s[4:5], 0x40
s_dcache_inv
v_mov_b32_e32 v0, v1
v_mov_b32_e32 v0, s1
v_mov_b32_e32 v0, 0
v_mov_b32_e32 v0, -1
v_mov_b32_e32 v0, 64
v_mov_b32_e32 v0, -16
v_mov_b32_e32 v0, 1.0
v_mov_b32_e32 v0, -2.0
v_mov_b32_e32 v0, 0.5
v_mov_b32_e32 v0, 4.0
v_mov_b32_e32 v0, 0.15915494
v_mov_b32_e32 v0, 0x3fb8aa3b
v_mov_b32_e32 v0, 0x7fffffff
v_mov_b32_e32 v0, 0x80000000
v_mov_b32_e32 v0, 0xff800000
v_mov_b32_e32 v255, v0
v_mov_b32_e32 v0, vcc_lo
v_mov_b32_e32 v0, exec_lo
v_mov_b32_e32 v0, m0
v_mov_b32_e64 v0, v1
v_readfirstlane_b32 s0, v1
v_readfirstlane_b32 s0, v255
v_cvt_f32_i32_e32 v0, v1
v_cvt_f32_u32_e32 v0, v1
v_cvt_u32_f32_e32 v0, v1
v_cvt_i32_f32_e32 v0, v1
v_cvt_f32_f16_e32 v0, v1
v_cvt_f32_f64_e32 v0, v[2:3]
v_cvt_f64_f32_e32 v[0:1], v2
v_rndne_f32_e32 v0, v1
v_floor_f32_e32 v0, v1
v_exp_f32_e32 v0, v1
v_exp_f32_e32 v0, s1
v_log_f32_e32 v0, v1
v_rcp_f32_e32 v0, v1
v_rcp_iflag_f32_e32 v0, v1
v_rsq_f32_e32 v0, v1
v_sqrt_f32_e32 v0, v1
v_add_f32_e32 v0, v1, v2
v_add_f32_e32 v0, s1, v2
v_add_f32_e32 v0, 1.0, v2
v_add_f32_e32 v0, -4.0, v2
v_add_f32_e32 v0, 0x3f000001, v2
v_add_f32_e32 v0, 0.15915494, v2
v_sub_f32_e32 v0, v1, v2
v_subrev_f32_e32 v0, v1, v2
v_mul_f32_e32 v0, v1, v2
v_mul_f32_e32 v0, 0x3fb8aa3b, v1
v_mul_f32_e32 v0, s1, v2
v_max_f32_e32 v0, v1, v2
v_max_f32_e32 v0, 0, v2
v_min_f32_e32 v0, v1, v2
v_max_i32_e32 v0, v1, v2
v_min_i32_e32 v0, v1, v2
v_max_u32_e32 v0, v1, v2
v_min_u32_e32 v0, v1, v2
v_mul_u32_u24_e32 v0, v1, v2
v_mul_u32_u24_e32 v0, s1, v2
v_lshrrev_b32_e32 v0, 2, v1
v_lshrrev_b32_e32 v0, v1, v2
v_ashrrev_i32_e32 v0, 31, v1
v_lshlrev_b32_e32 v0, 2, v1
v_lshlrev_b32_e32 v0, 4, v1
v_lshlrev_b32_e32 v0, s1, v2
v_and_b32_e32 v0, v1, v2
v_and_b32_e32 v0, 63, v1
v_and_b32_e32 v0, 0xffff, v1
v_and_b32_e32 v0, 0x7fffffff, v1
v_or_b32_e32 v0, v1, v2
v_xor_b32_e32 v0, v1, v2
v_xor_b32_e32 v0, 0x80000000, v1
v_add_nc_u32_e32 v0, v1, v2
v_add_nc_u32_e32 v0, s1, v2
v_add_nc_u32_e32 v0, 16, v2
v_sub_nc_u32_e32 v0, v1, v2
v_subrev_nc_u32_e32 v0, v1, v2
v_cndmask_b32_e32 v0, v1, v2, vcc
v_cndmask_b32_e32 v0, v1, v2, vcc_lo
v_cndmask_b32_e32 v0, 0, v2, vcc
v_cndmask_b32_e32 v0, 0, v2, vcc_lo
v_cndmask_b32_e32 v0, 1.0, v2, vcc
v_cndmask_b32_e32 v0, 1.0, v2, vcc_lo
v_ldexp_f32 v0, v1, v2
v_cmp_lt_f32_e32 vcc, v1, v2
v_cmp_lt_f32_e32 vcc_lo, v1, v2
v_cmp_eq_f32_e32 vcc, v1, v2
v_cmp_eq_f32_e32 vcc_lo, v1, v2
v_cmp_le_f32_e32 vcc, v1, v2
v_cmp_le_f32_e32 vcc_lo, v1, v2
v_cmp_gt_f32_e32 vcc, v1, v2
v_cmp_gt_f32_e32 vcc_lo, v1, v2
v_cmp_ge_f32_e32 vcc, v1, v2
v_cmp_ge_f32_e32 vcc_lo, v1, v2
v_cmp_u_f32_e32 vcc, v1, v1
v_cmp_u_f32_e32 vcc_lo, v1, v1
v_cmp_neq_f32_e32 vcc, v1, v2
v_cmp_neq_f32_e32 vcc_lo, v1, v2
v_cmp_lt_f32_e32 vcc, 0, v2
v_cmp_lt_f32_e32 vcc_lo, 0, v2
v_cmp_gt_f32_e32 vcc, s1, v2
v_cmp_gt_f32_e32 vcc_lo, s1, v2
v_cmp_lt_f32_e32 vcc, 0.5, v2
v_cmp_lt_f32_e32 vcc_lo, 0.5, v2
v_cmp_eq_u32_e32 vcc, v1, v2
v_cmp_eq_u32_e32 vcc_lo, v1, v2
v_cmp_ne_u32_e32 vcc, 0, v2
v_cmp_ne_u32_e32 vcc_lo, 0, v2
v_cmp_lt_u32_e32 vcc, v1, v2
v_cmp_lt_u32_e32 vcc_lo, v1, v2
v_cmp_gt_u32_e32 vcc, s1, v2
v_cmp_gt_u32_e32 vcc_lo, s1, v2
v_cmp_le_u32_e32 vcc, v1, v2
v_cmp_le_u32_e32 vcc_lo, v1, v2
v_cmp_ge_u32_e32 vcc, v1, v2
v_cmp_ge_u32_e32 vcc_lo, v1, v2
v_cmp_lt_i32_e32 vcc, v1, v2
v_cmp_lt_i32_e32 vcc_lo, v1, v2
v_cmp_gt_i32_e32 vcc, 0, v2
v_cmp_gt_i32_e32 vcc_lo, 0, v2
v_cmp_class_f32_e32 vcc, v1, v2
v_cmp_class_f32_e32 vcc_lo, v1, v2
v_cmp_class_f32_e64 vcc, v1, 3
v_cmp_class_f32_e64 vcc_lo, v1, 3
v_cmp_class_f32_e64 s[0:1], v1, s2
v_cmp_class_f32_e64 s0, v1, s2
v_cmp_lt_f32_e64 s[0:1], v1, v2
v_cmp_lt_f32_e64 s0, v1, v2
v_cmp_lt_f32_e64 s[0:1], v1, s2
v_cmp_lt_f32_e64 s0, v1, s2
v_cmp_lt_f32_e64 s[0:1], v1, 1.0
v_cmp_lt_f32_e64 s0, v1, 1.0
v_cmp_lt_f32_e64 s[0:1], |v1|, v2
v_cmp_lt_f32_e64 s0, |v1|, v2
v_cmp_lt_f32_e64 s[0:1], -v1, v2
v_cmp_lt_f32_e64 s0, -v1, v2
v_cmp_gt_u32_e64 s[0:1], s2, v1
v_cmp_gt_u32_e64 s0, s2, v1
v_cmp_eq_u32_e64 s[2:3], v1, 0
v_cmp_eq_u32_e64 s2, v1, 0
v_cmp_ne_u32_e64 vcc, v1, s2
v_cmp_ne_u32_e64 vcc_lo, v1, s2
v_cmp_eq_u32_e64 s[0:1], s2, s2
v_cmp_eq_u32_e64 s0, s2, s2
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
v_cvt_f32_i32_e64 v0, v1 div:2
v_fma_f32 v0, v1, v2, v3
v_fma_f32 v0, v1, s2, v3
v_fma_f32 v0, v1, v2, 1.0
v_fma_f32 v0, -v1, v2, v3
v_fma_f32 v0, v1, v2, -v3
v_fma_f32 v0, |v1|, v2, v3
v_fma_f32 v0, v1, v2, v3 clamp
v_fma_f32 v0, v1, 0.5, 0.5
v_add_f64 v[0:1], v[2:3], v[4:5]
v_add_f64 v[0:1], -v[2:3], v[4:5]
v_ldexp_f32 v0, v1, s2
v_mul_lo_u32 v0, v1, v2
v_mul_lo_u32 v0, s1, v2
v_mul_hi_u32 v0, v1, v2
v_add_nc_u32_e64 v0, v1, v2 clamp
v_add3_u32 v0, v1, v2, v3
v_add3_u32 v0, v1, s2, 4
v_lshl_add_u32 v0, v1, 2, v2
v_lshl_add_u32 v0, v1, 2, s2
v_lshl_or_b32 v0, v1, 16, v2
v_or3_b32 v0, v1, v2, v3
v_bfe_u32 v0, v1, 8, 8
v_bfe_u32 v0, v1, v2, v3
v_bfe_i32 v0, v1, 0, 16
v_cndmask_b32_e64 v0, v1, v2, s[0:1]
v_cndmask_b32_e64 v0, v1, v2, s0
v_cndmask_b32_e64 v0, 0, 1.0, vcc
v_cndmask_b32_e64 v0, 0, 1.0, vcc_lo
v_cndmask_b32_e64 v0, -v1, v2, s[0:1]
v_cndmask_b32_e64 v0, -v1, v2, s0
v_med3_f32 v0, v1, v2, v3
v_med3_f32 v0, v1, 0, 1.0
v_div_scale_f32 v0, vcc, v1, v2, v1
v_div_scale_f32 v0, vcc_lo, v1, v2, v1
v_div_scale_f32 v0, vcc, v2, v2, v1
v_div_scale_f32 v0, vcc_lo, v2, v2, v1
v_div_scale_f32 v0, s[2:3], v1, v2, v1
v_div_fmas_f32 v0, v1, v2, v3
v_div_fixup_f32 v0, v1, v2, v3
v_readlane_b32 s0, v1, 5
v_readlane_b32 s0, v1, 63
v_readlane_b32 s0, v1, s2
v_readlane_b32 s0, v1, m0
ds_load_b32 v0, v1
ds_load_b32 v0, v1 offset:4
ds_load_b32 v0, v1 offset:65535
ds_load_b128 v[0:3], v4
ds_load_b128 v[0:3], v4 offset:512
ds_store_b32 v1, v0
ds_store_b32 v1, v0 offset:4
ds_bpermute_b32 v0, v1, v2
ds_bpermute_b32 v0, v1, v2 offset:4
global_load_b32 v0, v[2:3], off
global_load_b32 v0, v[2:3], off offset:16
global_load_b32 v0, v[2:3], off offset:-16
global_load_b32 v0, v[2:3], off offset:4095
global_load_b32 v0, v[2:3], off offset:-4096
global_load_b32 v0, v1, s[2:3]
global_load_b32 v0, v1, s[2:3] offset:64
global_load_b32 v0, v1, s[2:3] offset:-64
global_load_b32 v0, v1, s[40:41] offset:4
global_load_b32 v0, v[2:3], off glc
global_load_b32 v0, v[2:3], off slc
global_load_b32 v0, v[2:3], off glc slc
global_load_b128 v[0:3], v[4:5], off
global_load_b128 v[0:3], v4, s[6:7] offset:16
global_load_b128 v[4:7], v0, s[2:3]
global_load_u8 v0, v[2:3], off
global_load_u8 v0, v1, s[2:3] offset:1
global_store_b32 v[2:3], v0, off
global_store_b32 v[2:3], v0, off offset:-4
global_store_b32 v1, v0, s[2:3]
global_store_b32 v1, v0, s[2:3] offset:4
global_store_b32 v[2:3], v0, off glc
global_store_b128 v[4:5], v[0:3], off
global_store_b128 v4, v[0:3], s[2:3] offset:16
global_store_b8 v[2:3], v0, off
global_store_b8 v1, v0, s[2:3] offset:3
global_store_b16 v[2:3], v0, off
global_store_b16 v1, v0, s[2:3] offset:6
global_atomic_add_u32 v[2:3], v0, off
global_atomic_add_u32 v1, v0, s[2:3]
global_atomic_add_u32 v0, v[2:3], v1, off glc
v_cmp_gt_u32_e32 vcc, s2, v0
v_cmp_gt_u32_e32 vcc_lo, s2, v0
s_and_saveexec_b64 s[4:5], vcc
s_and_saveexec_b32 s4, vcc_lo
s_cbranch_execz if_end
v_mov_b32_e32 v1, 1.0
s_or_b64 exec, exec, s[4:5]
if_end:
s_or_b64 exec, exec, s[4:5]
v_cmp_lt_f32_e32 vcc, 0, v1
v_cmp_lt_f32_e32 vcc_lo, 0, v1
s_and_saveexec_b64 s[6:7], vcc
s_and_saveexec_b32 s6, vcc_lo
s_xor_b64 s[6:7], exec, s[6:7]
s_cbranch_execz else_part
v_mov_b32_e32 v2, 1.0
else_part:
s_cbranch_execz else_end
v_mov_b32_e32 v2, 0
else_end:
s_or_b64 exec, exec, s[6:7]
s_mov_b64 s[8:9], exec
lane_loop:
v_add_nc_u32_e32 v3, 1, v3
v_cmp_ge_u32_e32 vcc, v3, v4
v_cmp_ge_u32_e32 vcc_lo, v3, v4
s_and_not1_b64 exec, exec, vcc
s_cbranch_execnz lane_loop
s_mov_b64 exec, s[8:9]
s_mov_b64 exec, -1
s_mov_b32 exec_lo, -1
s_mov_b32 exec_hi, 0
s_mov_b64 vcc, 0
s_mov_b64 vcc, exec
s_mov_b64 vcc, s[0:1]
s_and_b64 exec, exec, s[0:1]
s_and_not1_b64 exec, exec, s[0:1]
s_xor_b64 exec, exec, -1
s_cmp_eq_u64 exec, 0
s_cbranch_scc1 if_end
s_and_b64 vcc, exec, vcc
s_cbranch_vccz else_end
s_cbranch_vccnz lane_loop
s_cbranch_scc0 lane_loop
s_cbranch_execnz else_part
v_cvt_f64_f32_e32 v[1:2], v3
s_endpgm
s_waitcnt_vscnt null, 0x0
s_waitcnt_vscnt null, 0x3f
global_load_b32 v0, v[2:3], off glc dlc
global_load_b32 v0, v1, s[2:3] offset:-64 glc dlc
s_delay_alu instid0(VALU_DEP_1)
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(SALU_CYCLE_1)
s_delay_alu instid0(TRANS32_DEP_1)
s_delay_alu instid0(VALU_DEP_4)
s_waitcnt_depctr 0xffe3
s_waitcnt_depctr 0xfffe
s_waitcnt_depctr 0xfff
s_mov_b32 null, 0
buffer_gl0_inv
buffer_gl1_inv
