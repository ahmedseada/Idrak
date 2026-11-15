// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Cuda;

// The experiment's fused epilogues for the few-row products of stored weights (TypedGemvOutputs and SuperGemv variants):
// the product's blocks write their outputs, then the last block to finish (an atomic count of the blocks, reset by it for
// the next launch) runs the pass that would have followed as a kernel of its own, over the outputs as the other blocks
// left them (read past L1): the gate/up pair's act(gate) · up (Gated: SiLU, tanh GELU or ReLU by p_act), or residual + y
// and its RMS normalization (AddNorm: each row's sum of squares by the block's threads in a fixed order, then
// 1 / sqrt(sum / n + eps) and value · scale · (gain + offset), as RmsNormAffine computes it). Every thread reaches the
// kernels' DONE label, so the epilogue's barriers see the whole block.
internal static partial class PtxKernels
{
    /// <summary>A few-row product's fused epilogue.</summary>
    internal enum Epilogue
    {
        None,
        Gated,
        AddNorm,
    }

    /// <summary>The kernel name's suffix of an epilogue.</summary>
    internal static string EpilogueSuffix(Epilogue e) => e switch { Epilogue.Gated => "_gated", Epilogue.AddNorm => "_addnorm", _ => "" };

    /// <summary>The parameters an epilogue adds after the kernel's own.</summary>
    internal static string EpilogueParameters(Epilogue e) => e switch
    {
        Epilogue.Gated => ", .param .u64 p_h, .param .u64 p_count, .param .u32 p_act",
        Epilogue.AddNorm => ", .param .u64 p_r, .param .u64 p_sum, .param .u64 p_gain, .param .u64 p_out, .param .f32 p_eps, .param .f32 p_offset, .param .u64 p_count",
        _ => "",
    };

    /// <summary>The registers and shared memory an epilogue uses (declared with the kernel's).</summary>
    internal const string EpilogueRegisters =
        ".reg .b32 %e<16>; .reg .b64 %ed<12>; .reg .f32 %ef<16>; .reg .pred %ep<8>; .shared .align 4 .u32 ep_last; .shared .align 4 .f32 ep_red[256];";

    /// <summary>The epilogue's code, from the kernel's DONE label to just before its ret (the kernel's block: 256 threads).</summary>
    internal static string EpilogueCode(Epilogue e)
    {
        if (e == Epilogue.None)
        {
            return "";
        }

        const string head = """
                mov.u32 %e0, %tid.x;
                and.b32 %e14, %e0, 31;
                setp.eq.u32 %ep6, %e14, 0;
                @%ep6 membar.gl;
                bar.sync 0;
                setp.eq.u32 %ep0, %e0, 0;
                @!%ep0 bra EP_WAIT;
                ld.param.u64 %ed0, [p_count];
                cvta.to.global.u64 %ed0, %ed0;
                atom.global.add.u32 %e1, [%ed0], 1;
                mov.u32 %e2, %nctaid.x;
                sub.u32 %e2, %e2, 1;
                setp.eq.u32 %ep1, %e1, %e2;
                selp.u32 %e3, 1, 0, %ep1;
                st.shared.u32 [ep_last], %e3;
            EP_WAIT:
                bar.sync 0;
                ld.shared.u32 %e3, [ep_last];
                setp.eq.u32 %ep2, %e3, 0;
                @%ep2 bra EP_END;
                membar.gl;
                mov.u32 %e15, %ntid.x;
                ld.param.u32 %e4, [p_m];
                ld.param.u32 %e5, [p_n];

            """;
        const string tail = """
            EP_FIN:
                @!%ep0 bra EP_END;
                ld.param.u64 %ed0, [p_count];
                cvta.to.global.u64 %ed0, %ed0;
                st.global.u32 [%ed0], 0;
            EP_END:

            """;
        return head + (e == Epilogue.Gated ? Gated : AddNorm()) + tail;
    }

    // hidden = act(gate) · up over the pair's m · n outputs (gate p_y, up p_y1), act by p_act: 0 SiLU g / (1 + e^-g),
    // 1 tanh GELU 0.5 g (1 + tanh(√(2/π) (g + 0.044715 g³))), 2 ReLU.
    private const string Gated = """
            ld.param.u64 %ed1, [p_y];
            cvta.to.global.u64 %ed1, %ed1;
            ld.param.u64 %ed2, [p_y1];
            cvta.to.global.u64 %ed2, %ed2;
            ld.param.u64 %ed3, [p_h];
            cvta.to.global.u64 %ed3, %ed3;
            ld.param.u32 %e8, [p_act];
            mul.lo.u32 %e6, %e4, %e5;
            mov.u32 %e7, %e0;
        EP_G:
            setp.ge.u32 %ep3, %e7, %e6;
            @%ep3 bra EP_FIN;
            mul.wide.u32 %ed4, %e7, 4;
            add.u64 %ed5, %ed1, %ed4;
            ld.global.cg.f32 %ef0, [%ed5];
            add.u64 %ed5, %ed2, %ed4;
            ld.global.cg.f32 %ef1, [%ed5];
            setp.eq.u32 %ep4, %e8, 2;
            @%ep4 max.f32 %ef2, %ef0, 0f00000000;
            @%ep4 bra EP_ACT;
            setp.eq.u32 %ep4, %e8, 1;
            @%ep4 bra EP_GELU;
            mul.rn.f32 %ef3, %ef0, 0fBFB8AA3B;
            ex2.approx.f32 %ef3, %ef3;
            add.rn.f32 %ef3, %ef3, 0f3F800000;
            div.rn.f32 %ef2, %ef0, %ef3;
            bra EP_ACT;
        EP_GELU:
            mul.rn.f32 %ef3, %ef0, %ef0;
            mul.rn.f32 %ef3, %ef3, %ef0;
            fma.rn.f32 %ef3, %ef3, 0f3D372713, %ef0;
            mul.rn.f32 %ef3, %ef3, 0f3F4C422A;
            mul.rn.f32 %ef4, %ef3, 0f4038AA3B;
            ex2.approx.f32 %ef4, %ef4;
            add.rn.f32 %ef4, %ef4, 0f3F800000;
            div.rn.f32 %ef4, 0f40000000, %ef4;
            sub.rn.f32 %ef4, 0f3F800000, %ef4;
            add.rn.f32 %ef4, %ef4, 0f3F800000;
            mul.rn.f32 %ef4, %ef4, %ef0;
            mul.rn.f32 %ef2, %ef4, 0f3F000000;
        EP_ACT:
            mul.rn.f32 %ef5, %ef2, %ef1;
            add.u64 %ed5, %ed3, %ed4;
            st.global.f32 [%ed5], %ef5;
            add.u32 %e7, %e7, %e15;
            bra EP_G;

        """;

    // Per row: sum = residual + y (p_r, p_y into p_sum), its sum of squares by the threads (each its columns in order, then
    // the block's partial sums added pairwise in a fixed tree), scale = 1 / sqrt(sum / n + eps), out = sum · scale ·
    // (gain + offset).
    private static string AddNorm()
    {
        var tree = new System.Text.StringBuilder();
        for (int stride = 128; stride >= 1; stride /= 2)
        {
            tree.Append(System.Globalization.CultureInfo.InvariantCulture, $$"""
                    setp.lt.u32 %ep5, %e0, {{stride}};
                    @%ep5 ld.shared.f32 %ef7, [%e12+{{4 * stride}}];
                    @%ep5 ld.shared.f32 %ef8, [%e12];
                    @%ep5 add.rn.f32 %ef8, %ef8, %ef7;
                    @%ep5 st.shared.f32 [%e12], %ef8;
                    bar.sync 0;

                """);
        }

        return """
                ld.param.u64 %ed1, [p_y];
                cvta.to.global.u64 %ed1, %ed1;
                ld.param.u64 %ed2, [p_r];
                cvta.to.global.u64 %ed2, %ed2;
                ld.param.u64 %ed3, [p_sum];
                cvta.to.global.u64 %ed3, %ed3;
                ld.param.u64 %ed6, [p_gain];
                cvta.to.global.u64 %ed6, %ed6;
                ld.param.u64 %ed7, [p_out];
                cvta.to.global.u64 %ed7, %ed7;
                ld.param.f32 %ef10, [p_eps];
                ld.param.f32 %ef11, [p_offset];
                cvt.rn.f32.u32 %ef12, %e5;
                shl.b32 %e12, %e0, 2;
                mov.u32 %e13, ep_red;
                add.u32 %e12, %e12, %e13;
                mov.u32 %e9, 0;
            EP_ROW:
                setp.ge.u32 %ep3, %e9, %e4;
                @%ep3 bra EP_FIN;
                mul.lo.u32 %e10, %e9, %e5;
                mov.f32 %ef6, 0f00000000;
                mov.u32 %e7, %e0;
            EP_SUM:
                setp.ge.u32 %ep4, %e7, %e5;
                @%ep4 bra EP_SUMMED;
                add.u32 %e11, %e10, %e7;
                mul.wide.u32 %ed4, %e11, 4;
                add.u64 %ed5, %ed1, %ed4;
                ld.global.cg.f32 %ef0, [%ed5];
                add.u64 %ed5, %ed2, %ed4;
                ld.global.f32 %ef1, [%ed5];
                add.rn.f32 %ef2, %ef1, %ef0;
                add.u64 %ed5, %ed3, %ed4;
                st.global.f32 [%ed5], %ef2;
                fma.rn.f32 %ef6, %ef2, %ef2, %ef6;
                add.u32 %e7, %e7, %e15;
                bra EP_SUM;
            EP_SUMMED:
                st.shared.f32 [%e12], %ef6;
                bar.sync 0;

            """ + tree + """
                ld.shared.f32 %ef6, [ep_red];
                div.rn.f32 %ef6, %ef6, %ef12;
                add.rn.f32 %ef6, %ef6, %ef10;
                sqrt.rn.f32 %ef6, %ef6;
                div.rn.f32 %ef6, 0f3F800000, %ef6;
                mov.u32 %e7, %e0;
            EP_NORM:
                setp.ge.u32 %ep4, %e7, %e5;
                @%ep4 bra EP_NORMED;
                add.u32 %e11, %e10, %e7;
                mul.wide.u32 %ed4, %e11, 4;
                add.u64 %ed5, %ed3, %ed4;
                ld.global.f32 %ef2, [%ed5];
                mul.wide.u32 %ed8, %e7, 4;
                add.u64 %ed8, %ed6, %ed8;
                ld.global.f32 %ef3, [%ed8];
                add.rn.f32 %ef3, %ef3, %ef11;
                mul.rn.f32 %ef4, %ef2, %ef6;
                mul.rn.f32 %ef4, %ef4, %ef3;
                add.u64 %ed5, %ed7, %ed4;
                st.global.f32 [%ed5], %ef4;
                add.u32 %e7, %e7, %e15;
                bra EP_NORM;
            EP_NORMED:
                bar.sync 0;
                add.u32 %e9, %e9, 1;
                bra EP_ROW;

            """;
    }
}
