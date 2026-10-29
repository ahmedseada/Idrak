// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.PluginTests;

/// <summary>
/// The CUDA and HIP kernels the plug-in kernels ship (<see cref="PluginKernels"/>), the same two as the SPIR-V of
/// <see cref="PluginShaders"/>: for CUDA, what a plug-in gets from <c>nvcc -ptx</c> (written by hand here, in the style
/// of the library's own PTX); for HIP, GCN with virtual registers in the syntax of the library's own GCN kernels, which
/// Idrak assembles on each device (no SDK). Each loops over its n elements with a grid stride, so any number of blocks
/// works. The PTX targets sm_50 (the driver compiles it for newer GPUs) and is assembled with ptxas by a test when
/// ptxas is on the machine; the GCN is assembled for gfx900, gfx906, gfx908 and gfx90c and run in the GCN interpreter
/// against the CPU by the library's tests.
/// </summary>
internal static class PluginGpuSources
{
    /// <summary>
    /// One PTX module with two entries (both kernels share it on a device): <c>outside_unpack_pairs(packed, values, n)</c>,
    /// values[i] = the bfloat16 half i % 2 (low first) of word i / 2 of packed, and <c>outside_scale_rows(rows, scales,
    /// output, n, width)</c>, output[i] = rows[i] · scales[i / width].
    /// </summary>
    public const string Ptx = """
        .version 6.0
        .target sm_50
        .address_size 64

        // values[i] = the bfloat16 half i % 2 (low first) of word i / 2 of packed, as float32, for i < n.
        .visible .entry outside_unpack_pairs(
            .param .u64 .ptr .global .align 4 packed,
            .param .u64 .ptr .global .align 4 values,
            .param .u32 n
        )
        {
            .reg .pred %p<3>;
            .reg .b32 %r<12>;
            .reg .b64 %rd<7>;

            ld.param.u64 %rd1, [packed];
            ld.param.u64 %rd2, [values];
            ld.param.u32 %r1, [n];
            cvta.to.global.u64 %rd1, %rd1;
            cvta.to.global.u64 %rd2, %rd2;
            mov.u32 %r2, %ctaid.x;
            mov.u32 %r3, %ntid.x;
            mov.u32 %r4, %tid.x;
            mad.lo.u32 %r5, %r2, %r3, %r4;          // i
            mov.u32 %r6, %nctaid.x;
            mul.lo.u32 %r7, %r6, %r3;               // the grid's threads: the stride
        $L_unpack_next:
            setp.ge.u32 %p1, %r5, %r1;
            @%p1 bra $L_unpack_done;
            shr.u32 %r8, %r5, 1;
            mul.wide.u32 %rd3, %r8, 4;
            add.s64 %rd4, %rd1, %rd3;
            ld.global.u32 %r9, [%rd4];              // word i / 2
            and.b32 %r10, %r5, 1;
            setp.eq.u32 %p2, %r10, 0;
            shl.b32 %r11, %r9, 16;                  // low half
            and.b32 %r9, %r9, -65536;               // high half
            selp.b32 %r11, %r11, %r9, %p2;
            mul.wide.u32 %rd5, %r5, 4;
            add.s64 %rd6, %rd2, %rd5;
            st.global.u32 [%rd6], %r11;
            add.u32 %r5, %r5, %r7;
            bra.uni $L_unpack_next;
        $L_unpack_done:
            ret;
        }

        // output[i] = rows[i] * scales[i / width] for i < n.
        .visible .entry outside_scale_rows(
            .param .u64 .ptr .global .align 4 rows,
            .param .u64 .ptr .global .align 4 scales,
            .param .u64 .ptr .global .align 4 output,
            .param .u32 n,
            .param .u32 width
        )
        {
            .reg .pred %p<2>;
            .reg .b32 %r<10>;
            .reg .f32 %f<4>;
            .reg .b64 %rd<9>;

            ld.param.u64 %rd1, [rows];
            ld.param.u64 %rd2, [scales];
            ld.param.u64 %rd3, [output];
            ld.param.u32 %r1, [n];
            ld.param.u32 %r2, [width];
            cvta.to.global.u64 %rd1, %rd1;
            cvta.to.global.u64 %rd2, %rd2;
            cvta.to.global.u64 %rd3, %rd3;
            mov.u32 %r3, %ctaid.x;
            mov.u32 %r4, %ntid.x;
            mov.u32 %r5, %tid.x;
            mad.lo.u32 %r6, %r3, %r4, %r5;          // i
            mov.u32 %r7, %nctaid.x;
            mul.lo.u32 %r8, %r7, %r4;               // the stride
        $L_scale_next:
            setp.ge.u32 %p1, %r6, %r1;
            @%p1 bra $L_scale_done;
            mul.wide.u32 %rd4, %r6, 4;
            add.s64 %rd5, %rd1, %rd4;
            ld.global.f32 %f1, [%rd5];
            div.u32 %r9, %r6, %r2;                  // the row
            mul.wide.u32 %rd6, %r9, 4;
            add.s64 %rd7, %rd2, %rd6;
            ld.global.f32 %f2, [%rd7];
            mul.rn.f32 %f3, %f1, %f2;
            add.s64 %rd8, %rd3, %rd4;
            st.global.f32 [%rd8], %f3;
            add.u32 %r6, %r6, %r8;
            bra.uni $L_scale_next;
        $L_scale_done:
            ret;
        }
        """;

    /// <summary>
    /// <c>outside_unpack_pairs(packed, values, n)</c> in GCN, as the PTX's: values[i] = the bfloat16 half i % 2 (low first)
    /// of word i / 2 of packed, for i &lt; n, in blocks of 256, grid-stride by the dispatch's work-items (the dispatch
    /// packet's grid size).
    /// </summary>
    public const string GcnUnpackPairs = """
        .kernel outside_unpack_pairs
        .arg packed, 8, global
        .arg values, 8, global
        .arg n, 4, value
        .reqd_workgroup_size 256, 1, 1
        .user_sgpr dispatch_ptr
        .user_sgpr kernarg_segment_ptr
        .system_sgpr workgroup_id_x
          s_load_dword %s10, s[0:1], 0xc            ; the grid's work-items: the stride
          s_load_dwordx4 %s[0:3], s[2:3], 0x0       ; packed, values
          s_load_dword %s7, s[2:3], 0x10            ; n
          s_lshl_b32 %s8, s4, 8                     ; the block's first work-item
          v_add_u32_e32 %v0, %s8, v0                ; i
          v_cmp_gt_u32_e32 vcc, %s7, %v0
          s_and_saveexec_b64 %s[12:13], vcc         ; the work-items below n
          s_cbranch_execz .Lunpack_done
        .Lunpack_next:
          v_lshrrev_b32_e32 %v1, 1, %v0             ; word i / 2
          v_lshlrev_b32_e32 %v1, 2, %v1             ; its bytes
          global_load_dword %v2, %v1, %s[0:1]
          v_and_b32_e32 %v3, 1, %v0
          v_cmp_eq_u32_e32 vcc, 0, %v3              ; an even i takes the low half
          v_lshlrev_b32_e32 %v4, 16, %v2            ; the low half
          v_and_b32_e32 %v5, 0xffff0000, %v2        ; the high half
          v_cndmask_b32_e32 %v6, %v5, %v4, vcc
          v_lshlrev_b32_e32 %v7, 2, %v0
          global_store_dword %v7, %v6, %s[2:3]
          v_add_u32_e32 %v0, %s10, %v0              ; the next i, a grid further
          v_cmp_gt_u32_e32 vcc, %s7, %v0
          s_and_b64 exec, exec, vcc
          s_cbranch_execnz .Lunpack_next
        .Lunpack_done:
          s_endpgm
        """;

    /// <summary>
    /// <c>outside_scale_rows(rows, scales, output, n, width)</c> in GCN, as the PTX's: output[i] = rows[i] · scales[i / width]
    /// for i &lt; n, in blocks of 256, grid-stride; i / width as LLVM divides unsigned integers (a reciprocal of width
    /// once, then a high product and two corrections an element).
    /// </summary>
    public const string GcnScaleRows = """
        .kernel outside_scale_rows
        .arg rows, 8, global
        .arg scales, 8, global
        .arg output, 8, global
        .arg n, 4, value
        .arg width, 4, value
        .reqd_workgroup_size 256, 1, 1
        .user_sgpr dispatch_ptr
        .user_sgpr kernarg_segment_ptr
        .system_sgpr workgroup_id_x
          s_load_dword %s10, s[0:1], 0xc            ; the grid's work-items: the stride
          s_load_dwordx8 %s[0:7], s[2:3], 0x0       ; rows, scales, output, n, width
          s_lshl_b32 %s8, s4, 8                     ; the block's first work-item
          v_add_u32_e32 %v0, %s8, v0                ; i
          v_cvt_f32_u32_e32 %v1, %s7                ; 2^32 / width, rounded down (LLVM's unsigned division)
          s_sub_i32 %s9, 0, %s7
          v_rcp_iflag_f32_e32 %v1, %v1
          v_mul_f32_e32 %v1, 0x4f7ffffe, %v1
          v_cvt_u32_f32_e32 %v1, %v1
          v_mul_lo_u32 %v2, %s9, %v1
          v_mul_hi_u32 %v2, %v1, %v2
          v_add_u32_e32 %v1, %v1, %v2
          v_cmp_gt_u32_e32 vcc, %s6, %v0
          s_and_saveexec_b64 %s[12:13], vcc         ; the work-items below n
          s_cbranch_execz .Lscale_done
        .Lscale_next:
          v_lshlrev_b32_e32 %v3, 2, %v0
          global_load_dword %v4, %v3, %s[0:1]       ; rows[i]
          v_mul_hi_u32 %v5, %v0, %v1                ; the row, i / width: an estimate
          v_mul_lo_u32 %v6, %v5, %s7
          v_sub_u32_e32 %v6, %v0, %v6               ; its remainder
          v_add_u32_e32 %v7, 1, %v5
          v_cmp_le_u32_e32 vcc, %s7, %v6
          v_cndmask_b32_e32 %v5, %v5, %v7, vcc
          v_subrev_u32_e32 %v8, %s7, %v6
          v_cndmask_b32_e32 %v6, %v6, %v8, vcc
          v_add_u32_e32 %v7, 1, %v5
          v_cmp_le_u32_e32 vcc, %s7, %v6
          v_cndmask_b32_e32 %v5, %v5, %v7, vcc      ; the row
          v_lshlrev_b32_e32 %v9, 2, %v5
          global_load_dword %v10, %v9, %s[2:3]      ; scales[row]
          v_mul_f32_e32 %v11, %v4, %v10
          global_store_dword %v3, %v11, %s[4:5]
          v_add_u32_e32 %v0, %s10, %v0              ; the next i, a grid further
          v_cmp_gt_u32_e32 vcc, %s6, %v0
          s_and_b64 exec, exec, vcc
          s_cbranch_execnz .Lscale_next
        .Lscale_done:
          s_endpgm
        """;
}
