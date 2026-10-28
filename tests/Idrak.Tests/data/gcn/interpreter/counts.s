; counts: out[lane] = in[lane] + in[lane + 64] (uint32), one wave: two scalar loads waited for with lgkmcnt(0), two
; vector loads, the first waited for with vmcnt(1) (loads complete in order), the second with vmcnt(0).
; Arguments: in at 0, out at 8; 16 bytes.
  s_load_dwordx2 s[4:5], s[0:1], 0x0
  s_load_dwordx2 s[6:7], s[0:1], 0x8
  s_waitcnt lgkmcnt(0)
  v_lshlrev_b32_e32 v1, 2, v0
  global_load_dword v2, v1, s[4:5]
  global_load_dword v3, v1, s[4:5] offset:256
  s_waitcnt vmcnt(1)
  v_mov_b32_e32 v4, v2
  s_waitcnt vmcnt(0)
  v_add_u32_e32 v4, v4, v3
  global_store_dword v1, v4, s[6:7]
  s_endpgm
