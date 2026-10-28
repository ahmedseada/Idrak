; reduce: out[g] = the sum of in[256 g .. 256 g + 255] (floats), a workgroup of 256 work-items (four waves) a sum:
; each work-item's value into LDS, then a tree in LDS (strides 128, 64, ..., 1), an s_barrier before each step.
; Arguments: in at 0, out at 8 (global pointers); 16 bytes. LDS: 1,024 bytes. s2 the workgroup id x.
  s_load_dwordx4 s[4:7], s[0:1], 0x0
  s_lshl_b32 s3, s2, 8
  v_add_u32_e32 v1, s3, v0
  v_lshlrev_b32_e32 v2, 2, v1
  s_waitcnt lgkmcnt(0)
  global_load_dword v3, v2, s[4:5]
  v_lshlrev_b32_e32 v4, 2, v0
  s_waitcnt vmcnt(0)
  ds_write_b32 v4, v3
  s_movk_i32 s8, 0x80
.Lloop:
  s_waitcnt lgkmcnt(0)
  s_barrier
  v_cmp_gt_u32_e32 vcc, s8, v0
  s_and_saveexec_b64 s[10:11], vcc
  s_cbranch_execz .Lskip
  s_lshl_b32 s9, s8, 2
  v_add_u32_e32 v5, s9, v4
  ds_read_b32 v6, v4
  ds_read_b32 v7, v5
  s_waitcnt lgkmcnt(0)
  v_add_f32_e32 v6, v6, v7
  ds_write_b32 v4, v6
.Lskip:
  s_or_b64 exec, exec, s[10:11]
  s_lshr_b32 s8, s8, 1
  s_cmp_lg_u32 s8, 0
  s_cbranch_scc1 .Lloop
  s_waitcnt lgkmcnt(0)
  s_barrier
  v_cmp_eq_u32_e32 vcc, 0, v0
  s_and_saveexec_b64 s[10:11], vcc
  s_cbranch_execz .Lend
  v_mov_b32_e32 v8, 0
  ds_read_b32 v9, v8
  s_lshl_b32 s12, s2, 2
  v_mov_b32_e32 v10, s12
  s_waitcnt lgkmcnt(0)
  global_store_dword v10, v9, s[6:7]
.Lend:
  s_endpgm
