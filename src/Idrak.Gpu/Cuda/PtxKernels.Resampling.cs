// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;

namespace Idrak.Gpu.Cuda;

// PTX for resampling planes of [height, width] (Backend.Interpolate2d, AdaptiveAvgPool, AdaptiveMaxPool and their
// gradients): a thread per output for the forward passes, which computes the input positions it reads as the CPU does
// (in float, rounded at every step: nearest min(floor(o · scale), size - 1) with the exact cases output = input and
// output = 2 · input; bilinear floor(s) and the next position with weights 1 - λ and λ), and a thread per input element
// for the gradients, which gathers over the outputs that read it, in output order (no atomics: the same bits every run).
internal static partial class PtxKernels
{
    /// <summary>The kernels of this file (in the main module).</summary>
    public static readonly string[] ResamplingNames =
        ["interpolate_f32", "interpolate_bwd_f32", "adaptive_avgpool_f32", "adaptive_avgpool_bwd_f32", "adaptive_maxpool_f32", "adaptive_maxpool_bwd_f32"];

    private static void BuildResampling(StringBuilder sb)
    {
        var interpolation = new[] { ("u32", "planes"), ("u32", "height"), ("u32", "width"), ("u32", "outheight"), ("u32", "outwidth"), ("u32", "mode"), ("u32", "align"),
            ("f32", "scaleh"), ("f32", "scalew") };
        var adaptive = new[] { ("u32", "planes"), ("u32", "height"), ("u32", "width"), ("u32", "outheight"), ("u32", "outwidth") };

        Elementwise(sb, "interpolate_f32", ["x", "y"], interpolation, $"""
            rem.u32 %r5, %i, %s_outwidth;
            div.u32 %r6, %i, %s_outwidth;
            rem.u32 %r7, %r6, %s_outheight;
            div.u32 %r8, %r6, %s_outheight;
            {AxisTaps("%r7", "%s_scaleh", "%s_height", "%s_outheight", "%r10", "%r11", "%f10", "%f11", "ROWS")}
            {AxisTaps("%r5", "%s_scalew", "%s_width", "%s_outwidth", "%r12", "%r13", "%f12", "%f13", "COLS")}
            mad.lo.u32 %r14, %r8, %s_height, %r10;
            mul.lo.u32 %r14, %r14, %s_width;
            mad.lo.u32 %r15, %r8, %s_height, %r11;
            mul.lo.u32 %r15, %r15, %s_width;
            add.u32 %r16, %r14, %r12;
            mul.wide.u32 %rd1, %r16, 4;
            add.u64 %rd1, %b_x, %rd1;
            ld.global.f32 %f1, [%rd1];
            setp.eq.u32 %p1, %s_mode, 0;
            @%p1 bra STORE;
            add.u32 %r16, %r14, %r13;
            mul.wide.u32 %rd1, %r16, 4;
            add.u64 %rd1, %b_x, %rd1;
            ld.global.f32 %f2, [%rd1];
            add.u32 %r16, %r15, %r12;
            mul.wide.u32 %rd1, %r16, 4;
            add.u64 %rd1, %b_x, %rd1;
            ld.global.f32 %f3, [%rd1];
            add.u32 %r16, %r15, %r13;
            mul.wide.u32 %rd1, %r16, 4;
            add.u64 %rd1, %b_x, %rd1;
            ld.global.f32 %f4, [%rd1];
            mul.rn.f32 %f5, %f12, %f1;
            mul.rn.f32 %f6, %f13, %f2;
            add.rn.f32 %f5, %f5, %f6;
            mul.rn.f32 %f6, %f12, %f3;
            mul.rn.f32 %f7, %f13, %f4;
            add.rn.f32 %f6, %f6, %f7;
            mul.rn.f32 %f5, %f10, %f5;
            mul.rn.f32 %f6, %f11, %f6;
            add.rn.f32 %f1, %f5, %f6;
            STORE:
            st.global.f32 [%a_y], %f1;
            """);

        Elementwise(sb, "interpolate_bwd_f32", ["dy", "dx"], interpolation, $"""
            rem.u32 %r5, %i, %s_width;
            div.u32 %r6, %i, %s_width;
            rem.u32 %r7, %r6, %s_height;
            div.u32 %r6, %r6, %s_height;
            {Readers("%r7", "%s_scaleh", "%s_outheight", "%r8", "%r9", "RR")}
            {Readers("%r5", "%s_scalew", "%s_outwidth", "%r14", "%r15", "RC")}
            ld.global.f32 %f1, [%a_dx];
            mov.u32 %r16, %r8;
            ROW:
            setp.ge.s32 %p1, %r16, %r9;
            @%p1 bra END;
            {AxisTaps("%r16", "%s_scaleh", "%s_height", "%s_outheight", "%r10", "%r11", "%f10", "%f11", "TR")}
            setp.ne.u32 %p2, %r10, %r7;
            setp.ne.and.u32 %p2, %r11, %r7, %p2;
            @%p2 bra ROW_NEXT;
            {WeightOn("%r10", "%r11", "%f10", "%f11", "%r7", "%f2")}
            mad.lo.u32 %r17, %r6, %s_outheight, %r16;
            mul.lo.u32 %r17, %r17, %s_outwidth;
            mov.u32 %r18, %r14;
            COL:
            setp.ge.s32 %p1, %r18, %r15;
            @%p1 bra ROW_NEXT;
            {AxisTaps("%r18", "%s_scalew", "%s_width", "%s_outwidth", "%r12", "%r13", "%f12", "%f13", "TC")}
            setp.ne.u32 %p2, %r12, %r5;
            setp.ne.and.u32 %p2, %r13, %r5, %p2;
            @%p2 bra COL_NEXT;
            {WeightOn("%r12", "%r13", "%f12", "%f13", "%r5", "%f3")}
            add.u32 %r19, %r17, %r18;
            mul.wide.u32 %rd1, %r19, 4;
            add.u64 %rd1, %b_dy, %rd1;
            ld.global.f32 %f4, [%rd1];
            mul.rn.f32 %f5, %f2, %f3;
            mul.rn.f32 %f5, %f5, %f4;
            add.rn.f32 %f1, %f1, %f5;
            COL_NEXT:
            add.u32 %r18, %r18, 1;
            bra COL;
            ROW_NEXT:
            add.u32 %r16, %r16, 1;
            bra ROW;
            END:
            st.global.f32 [%a_dx], %f1;
            """);

        Elementwise(sb, "adaptive_avgpool_f32", ["x", "y"], adaptive, $"""
            rem.u32 %r5, %i, %s_outwidth;
            div.u32 %r6, %i, %s_outwidth;
            rem.u32 %r7, %r6, %s_outheight;
            div.u32 %r8, %r6, %s_outheight;
            {AdaptiveWindow("%r7", "%s_height", "%s_outheight", "%r10", "%r11")}
            {AdaptiveWindow("%r5", "%s_width", "%s_outwidth", "%r12", "%r13")}
            mov.f32 %f1, {Zero};
            mov.u32 %r14, %r10;
            AR:
            setp.ge.u32 %p1, %r14, %r11;
            @%p1 bra AR_END;
            mad.lo.u32 %r15, %r8, %s_height, %r14;
            mul.lo.u32 %r15, %r15, %s_width;
            mov.u32 %r16, %r12;
            AC:
            setp.ge.u32 %p1, %r16, %r13;
            @%p1 bra AR_NEXT;
            add.u32 %r17, %r15, %r16;
            mul.wide.u32 %rd1, %r17, 4;
            add.u64 %rd1, %b_x, %rd1;
            ld.global.f32 %f2, [%rd1];
            add.rn.f32 %f1, %f1, %f2;
            add.u32 %r16, %r16, 1;
            bra AC;
            AR_NEXT:
            add.u32 %r14, %r14, 1;
            bra AR;
            AR_END:
            sub.u32 %r18, %r11, %r10;
            cvt.rn.f32.u32 %f3, %r18;
            div.rn.f32 %f1, %f1, %f3;
            sub.u32 %r18, %r13, %r12;
            cvt.rn.f32.u32 %f3, %r18;
            div.rn.f32 %f1, %f1, %f3;
            st.global.f32 [%a_y], %f1;
            """);

        Elementwise(sb, "adaptive_avgpool_bwd_f32", ["dy", "dx"], adaptive, $"""
            rem.u32 %r5, %i, %s_width;
            div.u32 %r6, %i, %s_width;
            rem.u32 %r7, %r6, %s_height;
            div.u32 %r8, %r6, %s_height;
            {AdaptiveCovers("%r7", "%s_height", "%s_outheight", "%r20", "%r21")}
            {AdaptiveCovers("%r5", "%s_width", "%s_outwidth", "%r22", "%r23")}
            ld.global.f32 %f1, [%a_dx];
            mov.u32 %r14, %r20;
            BR:
            setp.ge.u32 %p1, %r14, %r21;
            @%p1 bra BR_END;
            {AdaptiveWindow("%r14", "%s_height", "%s_outheight", "%r10", "%r11")}
            setp.lt.u32 %p2, %r7, %r10;
            setp.ge.or.u32 %p2, %r7, %r11, %p2;
            @%p2 bra BR_NEXT;
            sub.u32 %r18, %r11, %r10;
            cvt.rn.f32.u32 %f4, %r18;
            mad.lo.u32 %r15, %r8, %s_outheight, %r14;
            mul.lo.u32 %r15, %r15, %s_outwidth;
            mov.u32 %r16, %r22;
            BC:
            setp.ge.u32 %p1, %r16, %r23;
            @%p1 bra BR_NEXT;
            {AdaptiveWindow("%r16", "%s_width", "%s_outwidth", "%r12", "%r13")}
            setp.lt.u32 %p2, %r5, %r12;
            setp.ge.or.u32 %p2, %r5, %r13, %p2;
            @%p2 bra BC_NEXT;
            add.u32 %r17, %r15, %r16;
            mul.wide.u32 %rd1, %r17, 4;
            add.u64 %rd1, %b_dy, %rd1;
            ld.global.f32 %f2, [%rd1];
            div.rn.f32 %f2, %f2, %f4;
            sub.u32 %r18, %r13, %r12;
            cvt.rn.f32.u32 %f3, %r18;
            div.rn.f32 %f2, %f2, %f3;
            add.rn.f32 %f1, %f1, %f2;
            BC_NEXT:
            add.u32 %r16, %r16, 1;
            bra BC;
            BR_NEXT:
            add.u32 %r14, %r14, 1;
            bra BR;
            BR_END:
            st.global.f32 [%a_dx], %f1;
            """);

        // The largest value of each window and its flat input index (int bits); the first in row order wins, a NaN wins
        // (each later NaN too), as the CPU's.
        Elementwise(sb, "adaptive_maxpool_f32", ["x", "y", "argmax"], adaptive, $"""
            rem.u32 %r5, %i, %s_outwidth;
            div.u32 %r6, %i, %s_outwidth;
            rem.u32 %r7, %r6, %s_outheight;
            div.u32 %r8, %r6, %s_outheight;
            {AdaptiveWindow("%r7", "%s_height", "%s_outheight", "%r10", "%r11")}
            {AdaptiveWindow("%r5", "%s_width", "%s_outwidth", "%r12", "%r13")}
            mul.lo.u32 %r9, %s_height, %s_width;
            mul.lo.u32 %r9, %r9, %r8;
            mov.f32 %f1, {NegInf};
            mad.lo.u32 %r19, %r10, %s_width, %r12;
            mov.u32 %r14, %r10;
            MR:
            setp.ge.u32 %p1, %r14, %r11;
            @%p1 bra MR_END;
            mov.u32 %r16, %r12;
            MC:
            setp.ge.u32 %p1, %r16, %r13;
            @%p1 bra MR_NEXT;
            mad.lo.u32 %r17, %r14, %s_width, %r16;
            add.u32 %r18, %r9, %r17;
            mul.wide.u32 %rd1, %r18, 4;
            add.u64 %rd1, %b_x, %rd1;
            ld.global.f32 %f2, [%rd1];
            setp.gt.f32 %p2, %f2, %f1;
            testp.notanumber.f32 %p3, %f2;
            or.pred %p2, %p2, %p3;
            selp.f32 %f1, %f2, %f1, %p2;
            selp.u32 %r19, %r17, %r19, %p2;
            add.u32 %r16, %r16, 1;
            bra MC;
            MR_NEXT:
            add.u32 %r14, %r14, 1;
            bra MR;
            MR_END:
            st.global.f32 [%a_y], %f1;
            add.u32 %r19, %r9, %r19;
            st.global.u32 [%a_argmax], %r19;
            """);

        // dx[e] += dy[o] of each window covering element e whose argmax is e, windows in order.
        Elementwise(sb, "adaptive_maxpool_bwd_f32", ["dy", "argmax", "dx"], adaptive, $"""
            rem.u32 %r5, %i, %s_width;
            div.u32 %r6, %i, %s_width;
            rem.u32 %r7, %r6, %s_height;
            div.u32 %r8, %r6, %s_height;
            {AdaptiveCovers("%r7", "%s_height", "%s_outheight", "%r20", "%r21")}
            {AdaptiveCovers("%r5", "%s_width", "%s_outwidth", "%r22", "%r23")}
            ld.global.f32 %f1, [%a_dx];
            mov.u32 %r14, %r20;
            GR:
            setp.ge.u32 %p1, %r14, %r21;
            @%p1 bra GR_END;
            mad.lo.u32 %r15, %r8, %s_outheight, %r14;
            mul.lo.u32 %r15, %r15, %s_outwidth;
            mov.u32 %r16, %r22;
            GC:
            setp.ge.u32 %p1, %r16, %r23;
            @%p1 bra GR_NEXT;
            add.u32 %r17, %r15, %r16;
            mul.wide.u32 %rd1, %r17, 4;
            add.u64 %rd2, %b_argmax, %rd1;
            ld.global.u32 %r18, [%rd2];
            setp.ne.u32 %p2, %r18, %i;
            @%p2 bra GC_NEXT;
            add.u64 %rd3, %b_dy, %rd1;
            ld.global.f32 %f2, [%rd3];
            add.rn.f32 %f1, %f1, %f2;
            GC_NEXT:
            add.u32 %r16, %r16, 1;
            bra GC;
            GR_NEXT:
            add.u32 %r14, %r14, 1;
            bra GR;
            GR_END:
            st.global.f32 [%a_dx], %f1;
            """);
    }

    // The input positions output position `o` of one axis reads and their weights (mode 0 nearest, else bilinear), as the
    // CPU computes them. Scratch: %r28, %r29, %f28 … %f30, %p12, %p13.
    private static string AxisTaps(string o, string scale, string size, string outSize, string first, string second, string w0, string w1, string label) => $"""
        setp.eq.u32 %p12, %s_mode, 0;
        @!%p12 bra {label}_BILINEAR;
        mov.u32 {first}, {o};
        setp.eq.u32 %p13, {outSize}, {size};
        @%p13 bra {label}_NEAREST;
        shl.b32 %r28, {size}, 1;
        shr.u32 {first}, {o}, 1;
        setp.eq.u32 %p13, {outSize}, %r28;
        @%p13 bra {label}_NEAREST;
        cvt.rn.f32.u32 %f28, {o};
        mul.rn.f32 %f28, %f28, {scale};
        cvt.rmi.s32.f32 %r28, %f28;
        sub.u32 %r29, {size}, 1;
        min.s32 {first}, %r28, %r29;
        {label}_NEAREST:
        mov.u32 {second}, {first};
        mov.f32 {w0}, {One};
        mov.f32 {w1}, {Zero};
        bra {label}_END;
        {label}_BILINEAR:
        cvt.rn.f32.u32 %f28, {o};
        setp.ne.u32 %p13, %s_align, 0;
        @%p13 mul.rn.f32 %f29, {scale}, %f28;
        @!%p13 add.rn.f32 %f29, %f28, 0f3F000000;
        @!%p13 mul.rn.f32 %f29, {scale}, %f29;
        @!%p13 sub.rn.f32 %f29, %f29, 0f3F000000;
        @!%p13 max.f32 %f29, %f29, {Zero};
        cvt.rmi.s32.f32 %r28, %f29;
        sub.u32 %r29, {size}, 1;
        min.s32 {first}, %r28, %r29;
        cvt.rn.f32.s32 %f30, {first};
        sub.rn.f32 %f30, %f29, %f30;
        max.f32 %f30, %f30, {Zero};
        min.f32 %f30, %f30, {One};
        setp.lt.s32 %p13, {first}, %r29;
        selp.u32 %r28, 1, 0, %p13;
        add.u32 {second}, {first}, %r28;
        sub.rn.f32 {w0}, {One}, %f30;
        mov.f32 {w1}, %f30;
        {label}_END:
        """;

    // The weight taps (first, second, w0, w1) put on input position i into `weight` (nearest: w0 where first is i).
    private static string WeightOn(string first, string second, string w0, string w1, string i, string weight) => $"""
        setp.eq.u32 %p12, {first}, {i};
        selp.f32 {weight}, {w0}, {Zero}, %p12;
        setp.eq.u32 %p13, {second}, {i};
        setp.ne.and.u32 %p13, %s_mode, 0, %p13;
        @%p13 add.rn.f32 {weight}, {weight}, {w1};
        """;

    // The outputs [first, last) whose source position can lie within two of input position i (all when the scale is 0).
    private static string Readers(string i, string scale, string outSize, string first, string last, string label) => $"""
        mov.u32 {first}, 0;
        mov.u32 {last}, {outSize};
        setp.gt.f32 %p14, {scale}, {Zero};
        @!%p14 bra {label}_ALL;
        sub.s32 %r28, {i}, 2;
        cvt.rn.f32.s32 %f28, %r28;
        div.rn.f32 %f28, %f28, {scale};
        cvt.rmi.s32.f32 %r28, %f28;
        sub.s32 %r28, %r28, 2;
        max.s32 %r28, %r28, 0;
        min.s32 {first}, %r28, {outSize};
        add.u32 %r28, {i}, 2;
        cvt.rn.f32.u32 %f28, %r28;
        div.rn.f32 %f28, %f28, {scale};
        cvt.rmi.s32.f32 %r28, %f28;
        add.s32 %r28, %r28, 4;
        max.s32 %r28, %r28, 0;
        min.s32 {last}, %r28, {outSize};
        {label}_ALL:
        """;

    // Adaptive window o of an axis: [floor(o · size / outSize), ceil((o + 1) · size / outSize)).
    private static string AdaptiveWindow(string o, string size, string outSize, string start, string end) => $"""
        mul.lo.u32 {start}, {o}, {size};
        div.u32 {start}, {start}, {outSize};
        mad.lo.u32 {end}, {o}, {size}, {size};
        add.u32 {end}, {end}, {outSize};
        sub.u32 {end}, {end}, 1;
        div.u32 {end}, {end}, {outSize};
        """;

    // The adaptive windows that can cover input position i: [floor(i · outSize / size) - 1, ceil((i + 1) · outSize / size) + 1), clipped.
    private static string AdaptiveCovers(string i, string size, string outSize, string first, string last) => $"""
        mul.lo.u32 {first}, {i}, {outSize};
        div.u32 {first}, {first}, {size};
        sub.s32 {first}, {first}, 1;
        max.s32 {first}, {first}, 0;
        mad.lo.u32 {last}, {i}, {outSize}, {outSize};
        add.u32 {last}, {last}, {size};
        sub.u32 {last}, {last}, 1;
        div.u32 {last}, {last}, {size};
        add.u32 {last}, {last}, 1;
        min.u32 {last}, {last}, {outSize};
        """;
}
