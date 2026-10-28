; lanes: lane movement in one wave of 64, each result to out[k * 64 + lane] (uint32), from x = 3 lane + 1: DPP
; (row_shr with bound_ctrl, quad_perm, row_bcast under a row mask, wave_shr, row_mirror), ds_swizzle, ds_bpermute,
; v_readlane, v_writelane, v_mbcnt, v_readfirstlane under a narrowed EXEC, and SDWA selections.
; Arguments: out at 0; 8 bytes. s[0:1] the arguments.
  s_load_dwordx2 s[4:5], s[0:1], 0x0
  v_mad_u32_u24 v1, v0, 3, 1
  v_lshlrev_b32_e32 v20, 2, v0
  v_mov_b32_e32 v2, 0
  v_mov_b32_dpp v2, v1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
  v_mov_b32_e32 v3, -1
  v_mov_b32_dpp v3, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
  v_mov_b32_e32 v4, 7
  v_mov_b32_dpp v4, v1 row_bcast:15 row_mask:0xa bank_mask:0xf
  ds_swizzle_b32 v5, v1 offset:swizzle(SWAP,16)
  v_sub_u32_e32 v6, 63, v0
  v_lshlrev_b32_e32 v6, 2, v6
  ds_bpermute_b32 v7, v6, v1
  v_readlane_b32 s6, v1, 5
  v_mov_b32_e32 v8, s6
  v_mov_b32_e32 v9, 0
  s_movk_i32 s7, 0x2a
  v_writelane_b32 v9, s7, 7
  v_mbcnt_lo_u32_b32 v10, -1, 0
  v_mbcnt_hi_u32_b32 v10, -1, v10
  v_mov_b32_e32 v11, 9
  v_mov_b32_dpp v11, v1 wave_shr:1 row_mask:0xf bank_mask:0xf
  v_mov_b32_dpp v12, v1 row_mirror row_mask:0xf bank_mask:0xf
  s_mov_b64 s[10:11], exec
  s_mov_b32 exec_lo, 0xff00
  s_mov_b32 exec_hi, 0
  v_readfirstlane_b32 s8, v1
  s_mov_b64 exec, s[10:11]
  v_mov_b32_e32 v13, s8
  v_mov_b32_sdwa v14, v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:BYTE_0
  v_add_u32_sdwa v15, v1, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_0
  s_waitcnt lgkmcnt(0)
  global_store_dword v20, v2, s[4:5]
  global_store_dword v20, v3, s[4:5] offset:256
  global_store_dword v20, v4, s[4:5] offset:512
  global_store_dword v20, v5, s[4:5] offset:768
  global_store_dword v20, v7, s[4:5] offset:1024
  global_store_dword v20, v8, s[4:5] offset:1280
  global_store_dword v20, v9, s[4:5] offset:1536
  global_store_dword v20, v10, s[4:5] offset:1792
  global_store_dword v20, v11, s[4:5] offset:2048
  global_store_dword v20, v12, s[4:5] offset:2304
  global_store_dword v20, v13, s[4:5] offset:2560
  global_store_dword v20, v14, s[4:5] offset:2816
  global_store_dword v20, v15, s[4:5] offset:3072
  s_endpgm
