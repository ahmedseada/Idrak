; math: float32 arithmetic per lane of one wave, each result to out[k * 64 + lane], from a[lane] and b[lane]: a / b
; as LLVM divides (div_scale, rcp, fma refinement, div_fmas, div_fixup), the transcendentals, fma and mad, med3,
; clamp, conversions, rounding, a compare and cndmask, ldexp, and two halves packed toward zero.
; Arguments: a at 0, b at 8, out at 16; 24 bytes.
  s_load_dwordx4 s[4:7], s[0:1], 0x0
  s_load_dwordx2 s[8:9], s[0:1], 0x10
  v_lshlrev_b32_e32 v1, 2, v0
  s_waitcnt lgkmcnt(0)
  global_load_dword v2, v1, s[4:5]
  global_load_dword v3, v1, s[6:7]
  s_waitcnt vmcnt(0)
  v_div_scale_f32 v4, s[10:11], v3, v3, v2
  v_div_scale_f32 v5, vcc, v2, v3, v2
  v_rcp_f32_e32 v6, v4
  v_fma_f32 v7, -v4, v6, 1.0
  v_fma_f32 v6, v7, v6, v6
  v_mul_f32_e32 v7, v5, v6
  v_fma_f32 v8, -v4, v7, v5
  v_fma_f32 v7, v8, v6, v7
  v_fma_f32 v4, -v4, v7, v5
  v_div_fmas_f32 v4, v4, v6, v7
  v_div_fixup_f32 v4, v4, v3, v2
  v_exp_f32_e32 v5, v2
  v_log_f32_e64 v6, |v3|
  v_sin_f32_e32 v7, v2
  v_cos_f32_e32 v8, v2
  v_sqrt_f32_e64 v9, |v3|
  v_rsq_f32_e64 v10, |v3|
  v_fma_f32 v11, v2, v3, -v2
  v_mad_f32 v12, v2, v3, 0.5
  v_med3_f32 v13, v2, -1.0, 1.0
  v_max_f32_e64 v14, v2, v3 clamp
  v_cvt_i32_f32_e32 v15, v2
  v_cvt_f16_f32_e32 v16, v2
  v_cvt_f32_f16_e32 v16, v16
  v_rndne_f32_e32 v17, v2
  v_fract_f32_e32 v18, v2
  v_cmp_lt_f32_e32 vcc, v2, v3
  v_cndmask_b32_e64 v19, v3, v2, vcc
  v_ldexp_f32 v21, v2, 3
  v_cvt_pkrtz_f16_f32 v22, v2, v3
  global_store_dword v1, v4, s[8:9]
  global_store_dword v1, v5, s[8:9] offset:256
  global_store_dword v1, v6, s[8:9] offset:512
  global_store_dword v1, v7, s[8:9] offset:768
  global_store_dword v1, v8, s[8:9] offset:1024
  global_store_dword v1, v9, s[8:9] offset:1280
  global_store_dword v1, v10, s[8:9] offset:1536
  global_store_dword v1, v11, s[8:9] offset:1792
  global_store_dword v1, v12, s[8:9] offset:2048
  global_store_dword v1, v13, s[8:9] offset:2304
  global_store_dword v1, v14, s[8:9] offset:2560
  global_store_dword v1, v15, s[8:9] offset:2816
  global_store_dword v1, v16, s[8:9] offset:3072
  global_store_dword v1, v17, s[8:9] offset:3328
  global_store_dword v1, v18, s[8:9] offset:3584
  global_store_dword v1, v19, s[8:9] offset:3840
  s_add_u32 s8, s8, 0x1000
  s_addc_u32 s9, s9, 0
  global_store_dword v1, v21, s[8:9]
  global_store_dword v1, v22, s[8:9] offset:256
  s_endpgm
