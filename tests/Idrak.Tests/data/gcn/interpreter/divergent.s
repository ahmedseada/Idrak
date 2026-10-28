; divergent: out[i] = i odd ? 3 i : 100 - i for every i < count (a uint32 array), in workgroups of 128 work-items
; (two waves). EXEC is narrowed by the bounds check, then split by an if/else and joined again.
; Arguments: out (a global pointer) at 0, count (a uint32) at 8; 16 bytes. s[0:1] the arguments, s2 the workgroup id x.
  s_load_dwordx2 s[4:5], s[0:1], 0x0
  s_load_dword s6, s[0:1], 0x8
  s_lshl_b32 s2, s2, 7
  v_add_u32_e32 v1, s2, v0
  s_waitcnt lgkmcnt(0)
  v_cmp_gt_u32_e32 vcc, s6, v1
  s_and_saveexec_b64 s[8:9], vcc
  s_cbranch_execz .Lend
  v_and_b32_e32 v2, 1, v1
  v_cmp_ne_u32_e32 vcc, 0, v2
  s_and_saveexec_b64 s[10:11], vcc
  s_xor_b64 s[10:11], exec, s[10:11]
  s_cbranch_execz .Lelse
  v_mul_u32_u24_e32 v3, 3, v1
.Lelse:
  s_andn2_saveexec_b64 s[10:11], s[10:11]
  s_cbranch_execz .Lendif
  v_sub_u32_e32 v3, 0x64, v1
.Lendif:
  s_or_b64 exec, exec, s[10:11]
  v_lshlrev_b32_e32 v4, 2, v1
  global_store_dword v4, v3, s[4:5]
.Lend:
  s_endpgm
