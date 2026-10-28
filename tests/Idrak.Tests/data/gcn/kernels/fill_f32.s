; fill_f32: out[i] = value for every i < count. The gate kernel of plan 35 (G0c): Idrak's own code object for it
; must equal, byte for byte, the one tools/gcn/make_gcn_golden.py makes from this file with llvm-mc and ld.lld.
;
; Arguments (16 bytes): out, a global float pointer, at 0; value, a float, at 8; count, a uint32, at 12.
; Launch: blocks of 256 x 1 x 1 work-items (declared below: max_flat_workgroup_size and reqd_workgroup_size),
; ceil(count / 256) blocks. The byte offset is 32-bit and unsigned, so count is at most 2^30. (An argument
; named n would not do: LLVM's metadata reader takes a bare n for a boolean.)
;
; Registers on entry: s[0:1] the kernel arguments' address (the only user SGPRs), s2 the workgroup id x (the first
; system SGPR), v0 the work-item id x. The script assembles this one source as code object v4 and v5; for v5 it
; writes amdhsa.version [ 1, 2 ] in place of v4's [ 1, 1 ].
  .amdgcn_target "amdgcn-amd-amdhsa--gfx90c"
  .text
  .globl fill_f32
  .p2align 8
  .type fill_f32,@function
fill_f32:
  s_load_dwordx4 s[4:7], s[0:1], 0x0           ; s[4:5] out, s6 value, s7 count
  s_lshl_b32 s2, s2, 8                         ; the block's first element: workgroup id x * 256
  v_add_u32_e32 v0, s2, v0                     ; this work-item's element
  s_waitcnt lgkmcnt(0)
  v_cmp_gt_u32_e32 vcc, s7, v0                 ; element < count
  s_and_saveexec_b64 s[0:1], vcc
  s_cbranch_execz .Lfill_f32_done
  v_lshlrev_b32_e32 v0, 2, v0                  ; byte offset
  v_mov_b32_e32 v1, s6
  global_store_dword v0, v1, s[4:5]
.Lfill_f32_done:
  s_endpgm
.Lfill_f32_end:
  .size fill_f32, .Lfill_f32_end-fill_f32

  .rodata
  .p2align 6
  .amdhsa_kernel fill_f32
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 16
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_private_segment_buffer 0
    .amdhsa_user_sgpr_dispatch_ptr 0
    .amdhsa_user_sgpr_queue_ptr 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_user_sgpr_dispatch_id 0
    .amdhsa_user_sgpr_flat_scratch_init 0
    .amdhsa_user_sgpr_private_segment_size 0
    .amdhsa_system_sgpr_private_segment_wavefront_offset 0
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 0
    .amdhsa_system_sgpr_workgroup_id_z 0
    .amdhsa_system_sgpr_workgroup_info 0
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 2
    .amdhsa_next_free_sgpr 8
    .amdhsa_reserve_vcc 1
    .amdhsa_reserve_flat_scratch 0
    .amdhsa_reserve_xnack_mask 1
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
    .amdhsa_dx10_clamp 1
    .amdhsa_ieee_mode 1
    .amdhsa_fp16_overflow 0
    .amdhsa_exception_fp_ieee_invalid_op 0
    .amdhsa_exception_fp_denorm_src 0
    .amdhsa_exception_fp_ieee_div_zero 0
    .amdhsa_exception_fp_ieee_overflow 0
    .amdhsa_exception_fp_ieee_underflow 0
    .amdhsa_exception_fp_ieee_inexact 0
    .amdhsa_exception_int_div_zero 0
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 1 ]
amdhsa.target: amdgcn-amd-amdhsa--gfx90c
amdhsa.kernels:
  - .name: fill_f32
    .symbol: fill_f32.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 12
    .vgpr_count: 2
    .sgpr_spill_count: 0
    .vgpr_spill_count: 0
    .max_flat_workgroup_size: 256
    .reqd_workgroup_size: [ 256, 1, 1 ]
    .args:
      - { .name: out, .offset: 0, .size: 8, .value_kind: global_buffer, .address_space: global }
      - { .name: value, .offset: 8, .size: 4, .value_kind: by_value }
      - { .name: count, .offset: 12, .size: 4, .value_kind: by_value }
...
  .end_amdgpu_metadata
