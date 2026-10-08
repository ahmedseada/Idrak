// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.PluginTests;

/// <summary>
/// The CUDA and HIP kernels the plug-in kernels ship (<see cref="PluginKernels"/>), the same two as the SPIR-V of
/// <see cref="PluginShaders"/>: what a plug-in gets from <c>nvcc -ptx</c> (written by hand here, in the style of the
/// library's own PTX) and the HIP C++ hipRTC compiles on the device. Each loops over its n elements with a grid stride,
/// so any number of blocks works. The PTX targets sm_50 (the driver compiles it for newer GPUs) and is assembled with
/// ptxas by a test when ptxas is on the machine.
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

    /// <summary>The same two kernels in HIP C++, one source (both kernels share its module on a device).</summary>
    public const string Hip = """
        // values[i] = the bfloat16 half i % 2 (low first) of word i / 2 of packed, as float32, for i < n.
        extern "C" __global__ void outside_unpack_pairs(const unsigned int* packed, float* values, int n)
        {
            for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < (long long)n; i += (long long)blockDim.x * gridDim.x)
            {
                unsigned int word = packed[i >> 1];
                values[i] = __uint_as_float((i & 1) == 0 ? word << 16 : word & 0xFFFF0000u);
            }
        }

        // output[i] = rows[i] * scales[i / width] for i < n.
        extern "C" __global__ void outside_scale_rows(const float* rows, const float* scales, float* output, int n, int width)
        {
            for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < (long long)n; i += (long long)blockDim.x * gridDim.x)
            {
                output[i] = rows[i] * scales[i / width];
            }
        }
        """;
}
