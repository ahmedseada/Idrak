// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;

namespace Idrak.Gpu.Cuda;

// PTX for convolutions without unfolded patches, the depthwise kernels, average pooling and image resampling.
//
// The convolution and its two gradients are each a product whose operands are read where they lie (implicit products:
// the input's patches are gathered as the tile needs them, never written out), in blocks of 16 × 16 threads: one output
// a thread over 16 × 16 tiles ("_tile"), or 4 × 4 outputs a thread over 64 × 64 tiles, staged 16 terms of the sum at a
// time in shared memory (8.3 KB static); the CUDA backend measures which (or the composed path, or the depthwise kernels)
// is fastest per shape. The products, with batch entries along z (a loop past the grid's z limit):
//
//   forward          entry (image n, group q); rows: the group's filters; columns: output positions (oh, ow); the sum over
//                    (c, kh, kw) of weight · x[n, q·Cg + c, oh·SH - PH + kh·DH, ow·SW - PW + kw·DW]; then the bias and
//                    the activation (flags: 1 bias, activation · 2, as ConvActivation).
//   backward input   entry (n, q); rows: the group's channels; columns: input positions (ih, iw); the sum over (f, kh, kw)
//                    of weight · dy[n, q·Fg + f, oh, ow] where ih + PH - kh·DH = oh·SH for a whole oh in [0, OH) (and so
//                    for the width); added to dx.
//   backward weight  entry (q, split s); rows: the group's filters; columns: (c, kh, kw); the sum over split s's positions
//                    of dy · x; added to dweight with one split, else written to part[s] and added in split order by
//                    conv_split_reduce_f32.
//
// Every output adds its terms in the order of the sum (the shape alone fixes the bits). Parameters of every convolution
// kernel after its pointers: N, C, H, W, KH, KW, SH, SW, PH, PW, OH, OW, DH, DW, F, G, flags, splits.
internal static partial class PtxKernels
{
    /// <summary>The kernels of this file (in the main module).</summary>
    public static readonly string[] ConvolutionNames =
    [
        "conv_fwd_f32", "conv_fwd_tile_f32", "conv_bwd_input_f32", "conv_bwd_input_tile_f32", "conv_bwd_weight_f32", "conv_bwd_weight_tile_f32",
        "conv_split_reduce_f32", "conv_dw_f32", "conv_dw_bwd_input_f32", "conv_dw_bwd_weight_f32", "avgpool_f32", "avgpool_bwd_f32", "resize_normalize_f32",
    ];

    /// <summary>Threads of an implicit product's block (16 × 16), and the edges of its two tiles.</summary>
    public const int ConvThreads = 256, ConvTileEdge = 16, ConvBlockedEdge = 64;

    // The step along the sum and the padded row of the blocked tiles in shared memory (65 floats: the stores along the sum
    // and the reads along rows and columns fall in different banks).
    private const int ConvStep = 16, ConvStride = ConvBlockedEdge + 1;

    private static readonly string[] ConvScalars = ["N", "C", "H", "W", "KH", "KW", "SH", "SW", "PH", "PW", "OH", "OW", "DH", "DW", "F", "G", "flags", "splits"];

    private enum ConvPass
    {
        Forward,
        Input,
        Weight,
    }

    private static void BuildConvolution(StringBuilder sb)
    {
        foreach (var (pass, name) in new[] { (ConvPass.Forward, "conv_fwd"), (ConvPass.Input, "conv_bwd_input"), (ConvPass.Weight, "conv_bwd_weight") })
        {
            ConvProduct(sb, name + "_f32", pass, blocked: true);
            ConvProduct(sb, name + "_tile_f32", pass, blocked: false);
        }

        var conv = ConvScalars.Select(s => ("u32", s)).ToArray();

        // dw[i] += Σ_s part[s · count + i], splits in order.
        Elementwise(sb, "conv_split_reduce_f32", ["part", "dw"], [("u32", "count"), ("u32", "splits")], """
            ld.global.f32 %f1, [%a_dw];
            mov.u32 %r5, 0;
            mov.u32 %r6, %i;
            SPLIT:
            setp.ge.u32 %p1, %r5, %s_splits;
            @%p1 bra SPLIT_END;
            mul.wide.u32 %rd1, %r6, 4;
            add.u64 %rd1, %b_part, %rd1;
            ld.global.f32 %f2, [%rd1];
            add.f32 %f1, %f1, %f2;
            add.u32 %r6, %r6, %s_count;
            add.u32 %r5, %r5, 1;
            bra SPLIT;
            SPLIT_END:
            st.global.f32 [%a_dw], %f1;
            """);

        // Depthwise forward: a thread per output (n, f, oh, ow) adds its window, rows then columns, to the bias and applies
        // the activation; channel c = f / (F / C).
        Elementwise(sb, "conv_dw_f32", ["x", "w", "bias", "y"], conv, $"""
            rem.u32 %r5, %i, %s_OW;
            div.u32 %r6, %i, %s_OW;
            rem.u32 %r7, %r6, %s_OH;
            div.u32 %r8, %r6, %s_OH;
            rem.u32 %r9, %r8, %s_F;
            div.u32 %r10, %r8, %s_F;
            div.u32 %r11, %s_F, %s_C;
            div.u32 %r11, %r9, %r11;
            mad.lo.u32 %r12, %r10, %s_C, %r11;
            mul.lo.u32 %r12, %r12, %s_H;
            mul.lo.u32 %r13, %r9, %s_KH;
            mov.f32 %f1, {Zero};
            mov.u32 %r14, 0;
            DW_KH:
            setp.ge.u32 %p1, %r14, %s_KH;
            @%p1 bra DW_END;
            mul.lo.u32 %r15, %r14, %s_DH;
            mad.lo.u32 %r15, %r7, %s_SH, %r15;
            sub.u32 %r15, %r15, %s_PH;
            setp.ge.u32 %p2, %r15, %s_H;
            @%p2 bra DW_KH_NEXT;
            add.u32 %r16, %r12, %r15;
            mul.lo.u32 %r16, %r16, %s_W;
            add.u32 %r17, %r13, %r14;
            mul.lo.u32 %r17, %r17, %s_KW;
            mov.u32 %r18, 0;
            DW_KW:
            setp.ge.u32 %p1, %r18, %s_KW;
            @%p1 bra DW_KH_NEXT;
            mul.lo.u32 %r19, %r18, %s_DW;
            mad.lo.u32 %r19, %r5, %s_SW, %r19;
            sub.u32 %r19, %r19, %s_PW;
            setp.ge.u32 %p2, %r19, %s_W;
            @%p2 bra DW_KW_NEXT;
            add.u32 %r20, %r16, %r19;
            mul.wide.u32 %rd1, %r20, 4;
            add.u64 %rd1, %b_x, %rd1;
            ld.global.f32 %f2, [%rd1];
            add.u32 %r21, %r17, %r18;
            mul.wide.u32 %rd2, %r21, 4;
            add.u64 %rd2, %b_w, %rd2;
            ld.global.f32 %f3, [%rd2];
            fma.rn.f32 %f1, %f3, %f2, %f1;
            DW_KW_NEXT:
            add.u32 %r18, %r18, 1;
            bra DW_KW;
            DW_KH_NEXT:
            add.u32 %r14, %r14, 1;
            bra DW_KH;
            DW_END:
            and.b32 %r22, %s_flags, 1;
            setp.eq.u32 %p3, %r22, 0;
            @%p3 bra DW_ACT;
            mul.wide.u32 %rd3, %r9, 4;
            add.u64 %rd3, %b_bias, %rd3;
            ld.global.f32 %f4, [%rd3];
            add.f32 %f1, %f1, %f4;
            DW_ACT:
            shr.u32 %r23, %s_flags, 1;
            {Activation("%f1", "%r23", "DW")}
            st.global.f32 [%a_y], %f1;
            """);

        // Depthwise input gradient: a thread per input element (n, c, ih, iw) adds, over the channel's filters in order and
        // the windows covering the element in window order, weight · dy, to dx.
        Elementwise(sb, "conv_dw_bwd_input_f32", ["dy", "w", "dx"], conv, """
            rem.u32 %r5, %i, %s_W;
            div.u32 %r6, %i, %s_W;
            rem.u32 %r7, %r6, %s_H;
            div.u32 %r8, %r6, %s_H;
            rem.u32 %r9, %r8, %s_C;
            div.u32 %r10, %r8, %s_C;
            div.u32 %r11, %s_F, %s_C;
            ld.global.f32 %f1, [%a_dx];
            mov.u32 %r12, 0;
            DI_M:
            setp.ge.u32 %p1, %r12, %r11;
            @%p1 bra DI_END;
            mad.lo.u32 %r13, %r9, %r11, %r12;
            mad.lo.u32 %r14, %r10, %s_F, %r13;
            mul.lo.u32 %r14, %r14, %s_OH;
            mov.u32 %r15, 0;
            DI_KH:
            setp.ge.u32 %p1, %r15, %s_KH;
            @%p1 bra DI_M_NEXT;
            add.u32 %r16, %r7, %s_PH;
            mul.lo.u32 %r17, %r15, %s_DH;
            sub.u32 %r16, %r16, %r17;
            setp.lt.s32 %p2, %r16, 0;
            @%p2 bra DI_KH_NEXT;
            rem.u32 %r17, %r16, %s_SH;
            setp.ne.u32 %p2, %r17, 0;
            @%p2 bra DI_KH_NEXT;
            div.u32 %r17, %r16, %s_SH;
            setp.ge.u32 %p2, %r17, %s_OH;
            @%p2 bra DI_KH_NEXT;
            add.u32 %r18, %r14, %r17;
            mul.lo.u32 %r18, %r18, %s_OW;
            mad.lo.u32 %r19, %r13, %s_KH, %r15;
            mul.lo.u32 %r19, %r19, %s_KW;
            mov.u32 %r20, 0;
            DI_KW:
            setp.ge.u32 %p1, %r20, %s_KW;
            @%p1 bra DI_KH_NEXT;
            add.u32 %r21, %r5, %s_PW;
            mul.lo.u32 %r22, %r20, %s_DW;
            sub.u32 %r21, %r21, %r22;
            setp.lt.s32 %p2, %r21, 0;
            @%p2 bra DI_KW_NEXT;
            rem.u32 %r22, %r21, %s_SW;
            setp.ne.u32 %p2, %r22, 0;
            @%p2 bra DI_KW_NEXT;
            div.u32 %r22, %r21, %s_SW;
            setp.ge.u32 %p2, %r22, %s_OW;
            @%p2 bra DI_KW_NEXT;
            add.u32 %r23, %r18, %r22;
            mul.wide.u32 %rd1, %r23, 4;
            add.u64 %rd1, %b_dy, %rd1;
            ld.global.f32 %f2, [%rd1];
            add.u32 %r24, %r19, %r20;
            mul.wide.u32 %rd2, %r24, 4;
            add.u64 %rd2, %b_w, %rd2;
            ld.global.f32 %f3, [%rd2];
            fma.rn.f32 %f1, %f3, %f2, %f1;
            DI_KW_NEXT:
            add.u32 %r20, %r20, 1;
            bra DI_KW;
            DI_KH_NEXT:
            add.u32 %r15, %r15, 1;
            bra DI_KH;
            DI_M_NEXT:
            add.u32 %r12, %r12, 1;
            bra DI_M;
            DI_END:
            st.global.f32 [%a_dx], %f1;
            """);

        ConvDepthwiseWeight(sb);

        // Average pooling: a thread per output sums its window in row order and divides by KH·KW or (countpad 0) the
        // positions it covers; the gradient gathers dy / divisor of the windows covering each element, in window order.
        var pool = new[] { ("u32", "H"), ("u32", "W"), ("u32", "KH"), ("u32", "KW"), ("u32", "SH"), ("u32", "SW"), ("u32", "PH"), ("u32", "PW"),
            ("u32", "OH"), ("u32", "OW"), ("u32", "countpad") };
        Elementwise(sb, "avgpool_f32", ["x", "y"], pool, $"""
            rem.u32 %r5, %i, %s_OW;
            div.u32 %r6, %i, %s_OW;
            rem.u32 %r7, %r6, %s_OH;
            div.u32 %r8, %r6, %s_OH;
            mul.lo.u32 %r9, %s_H, %s_W;
            mul.lo.u32 %r9, %r9, %r8;
            {PoolWindow("%r7", "%r5")}
            mov.f32 %f1, {Zero};
            mov.u32 %r14, %r10;
            AP_R:
            setp.ge.s32 %p1, %r14, %r11;
            @%p1 bra AP_END;
            mad.lo.u32 %r15, %r14, %s_W, %r9;
            mov.u32 %r16, %r12;
            AP_C:
            setp.ge.s32 %p1, %r16, %r13;
            @%p1 bra AP_R_NEXT;
            add.u32 %r17, %r15, %r16;
            mul.wide.u32 %rd1, %r17, 4;
            add.u64 %rd1, %b_x, %rd1;
            ld.global.f32 %f2, [%rd1];
            add.f32 %f1, %f1, %f2;
            add.u32 %r16, %r16, 1;
            bra AP_C;
            AP_R_NEXT:
            add.u32 %r14, %r14, 1;
            bra AP_R;
            AP_END:
            {PoolDivisor("%f3")}
            div.rn.f32 %f1, %f1, %f3;
            st.global.f32 [%a_y], %f1;
            """);

        Elementwise(sb, "avgpool_bwd_f32", ["dy", "dx"], pool, $"""
            rem.u32 %r5, %i, %s_W;
            div.u32 %r6, %i, %s_W;
            rem.u32 %r7, %r6, %s_H;
            div.u32 %r8, %r6, %s_H;
            mul.lo.u32 %r9, %s_OH, %s_OW;
            mul.lo.u32 %r9, %r9, %r8;
            {Covering("%r7", "%s_PH", "%s_KH", "%s_SH", "%s_OH", "%r20", "%r21", "AB_H")}
            {Covering("%r5", "%s_PW", "%s_KW", "%s_SW", "%s_OW", "%r22", "%r23", "AB_W")}
            ld.global.f32 %f1, [%a_dx];
            mov.u32 %r24, %r20;
            AB_R:
            setp.ge.u32 %p1, %r24, %r21;
            @%p1 bra AB_END;
            mad.lo.u32 %r25, %r24, %s_OW, %r9;
            mov.u32 %r26, %r22;
            AB_C:
            setp.ge.u32 %p1, %r26, %r23;
            @%p1 bra AB_R_NEXT;
            add.u32 %r27, %r25, %r26;
            mul.wide.u32 %rd1, %r27, 4;
            add.u64 %rd1, %b_dy, %rd1;
            ld.global.f32 %f2, [%rd1];
            {PoolWindow("%r24", "%r26")}
            {PoolDivisor("%f3")}
            div.rn.f32 %f2, %f2, %f3;
            add.f32 %f1, %f1, %f2;
            add.u32 %r26, %r26, 1;
            bra AB_C;
            AB_R_NEXT:
            add.u32 %r24, %r24, 1;
            bra AB_R;
            AB_END:
            st.global.f32 [%a_dx], %f1;
            """);

        ResizeNormalize(sb);
    }

    // The window of output (oh, ow) clipped to the image: rows [%r10, %r11), columns [%r12, %r13) (signed).
    private static string PoolWindow(string oh, string ow) => $"""
        mul.lo.u32 %r10, {oh}, %s_SH;
        sub.s32 %r10, %r10, %s_PH;
        add.s32 %r11, %r10, %s_KH;
        min.s32 %r11, %r11, %s_H;
        max.s32 %r10, %r10, 0;
        mul.lo.u32 %r12, {ow}, %s_SW;
        sub.s32 %r12, %r12, %s_PW;
        add.s32 %r13, %r12, %s_KW;
        min.s32 %r13, %r13, %s_W;
        max.s32 %r12, %r12, 0;
        """;

    // The divisor of the window PoolWindow left in %r10 … %r13, into `target`: KH·KW, or the positions it covers (≥ 1).
    private static string PoolDivisor(string target) => $"""
        sub.s32 %r28, %r11, %r10;
        sub.s32 %r29, %r13, %r12;
        mul.lo.s32 %r28, %r28, %r29;
        max.s32 %r28, %r28, 1;
        setp.ne.u32 %p5, %s_countpad, 0;
        mul.lo.u32 %r29, %s_KH, %s_KW;
        selp.u32 %r28, %r29, %r28, %p5;
        cvt.rn.f32.u32 {target}, %r28;
        """;

    // The windows [first, last) along one axis whose span covers input coordinate i: o·stride - pad ≤ i < o·stride - pad + size.
    private static string Covering(string i, string pad, string size, string stride, string outputs, string first, string last, string label) => $"""
        add.u32 %r30, {i}, {pad};
        sub.s32 %r31, %r30, {size};
        add.s32 %r31, %r31, 1;
        mov.u32 {first}, 0;
        setp.le.s32 %p6, %r31, 0;
        @%p6 bra {label}_LOW;
        add.u32 %r31, %r31, {stride};
        sub.u32 %r31, %r31, 1;
        div.u32 {first}, %r31, {stride};
        {label}_LOW:
        div.u32 {last}, %r30, {stride};
        add.u32 {last}, {last}, 1;
        min.u32 {last}, {last}, {outputs};
        """;

    // The activation of code `code` (ConvActivation) applied to `v` in place, as the element-wise kernels compute each.
    private static string Activation(string v, string code, string label) => $"""
        setp.eq.u32 %p8, {code}, 1;
        @%p8 max.f32 {v}, {v}, {Zero};
        setp.eq.u32 %p8, {code}, 2;
        @!%p8 bra {label}_NOT_SIGMOID;
        mul.f32 %f20, {v}, {F(-1.4426950408889634f)};
        ex2.approx.ftz.f32 %f20, %f20;
        add.f32 %f20, %f20, {One};
        rcp.rn.f32 {v}, %f20;
        {label}_NOT_SIGMOID:
        setp.eq.u32 %p8, {code}, 3;
        @!%p8 bra {label}_NOT_TANH;
        mul.f32 %f20, {v}, {F(2.8853900817779268f)};
        ex2.approx.ftz.f32 %f20, %f20;
        add.f32 %f20, %f20, {One};
        rcp.rn.f32 %f20, %f20;
        fma.rn.f32 {v}, %f20, {F(-2f)}, {One};
        {label}_NOT_TANH:
        setp.eq.u32 %p8, {code}, 4;
        @!%p8 bra {label}_NOT_GELU;
        mul.f32 %f20, {v}, {v};
        mul.f32 %f21, %f20, {v};
        fma.rn.f32 %f21, %f21, {F(0.044715f)}, {v};
        mul.f32 %f21, %f21, {F(0.7978845608f)};
        mul.f32 %f21, %f21, {F(2.8853900817779268f)};
        ex2.approx.ftz.f32 %f21, %f21;
        add.f32 %f21, %f21, {One};
        rcp.rn.f32 %f21, %f21;
        fma.rn.f32 %f21, %f21, {F(-2f)}, {One};
        add.f32 %f21, %f21, {One};
        mul.f32 %f21, %f21, {v};
        mul.f32 {v}, %f21, {F(0.5f)};
        {label}_NOT_GELU:
        setp.eq.u32 %p8, {code}, 5;
        @!%p8 bra {label}_NOT_SILU;
        mul.f32 %f20, {v}, {F(-1.4426950408889634f)};
        ex2.approx.ftz.f32 %f20, %f20;
        add.f32 %f20, %f20, {One};
        rcp.rn.f32 %f20, %f20;
        mul.f32 {v}, {v}, %f20;
        {label}_NOT_SILU:
        """;

    // Depthwise weight gradient: a block per weight (f, kh, kw) sums dy · x over the positions (each thread a fixed
    // stride of them, then a fixed tree in shared memory) and adds the sum to dw.
    private static void ConvDepthwiseWeight(StringBuilder sb)
    {
        int threads = BlockSize;
        var p = new StringBuilder();
        p.AppendLine(".visible .entry conv_dw_bwd_weight_f32(");
        p.AppendLine("    .param .u64 p_x, .param .u64 p_dy, .param .u64 p_dw,");
        p.AppendLine(string.Join(",\n", ConvScalars.Select(s => $"    .param .u32 p_{s}")));
        p.AppendLine(")");
        p.AppendLine("{");
        p.AppendLine("    .reg .pred %p<8>;");
        p.AppendLine("    .reg .f32 %f<8>;");
        p.AppendLine("    .reg .b32 %r<40>;");
        p.AppendLine("    .reg .b64 %rd<8>;");
        p.AppendLine("    .reg .u32 " + string.Join(", ", ConvScalars.Select(s => $"%s_{s}")) + ";");
        p.AppendLine($"    .shared .align 4 .f32 red[{threads}];");
        foreach (string s in ConvScalars)
        {
            p.AppendLine($"    ld.param.u32 %s_{s}, [p_{s}];");
        }

        p.AppendLine($$"""
                ld.param.u64 %rd1, [p_x];
                cvta.to.global.u64 %rd1, %rd1;
                ld.param.u64 %rd2, [p_dy];
                cvta.to.global.u64 %rd2, %rd2;
                ld.param.u64 %rd3, [p_dw];
                cvta.to.global.u64 %rd3, %rd3;
                mov.u32 %r1, %ctaid.x;
                mov.u32 %r2, %tid.x;
                mov.u32 %r3, %ntid.x;
                rem.u32 %r4, %r1, %s_KW;
                div.u32 %r5, %r1, %s_KW;
                rem.u32 %r6, %r5, %s_KH;
                div.u32 %r7, %r5, %s_KH;
                div.u32 %r8, %s_F, %s_C;
                div.u32 %r8, %r7, %r8;
                mul.lo.u32 %r9, %s_OH, %s_OW;
                mul.lo.u32 %r10, %r9, %s_N;
                mul.lo.u32 %r11, %r6, %s_DH;
                sub.u32 %r11, %r11, %s_PH;
                mul.lo.u32 %r12, %r4, %s_DW;
                sub.u32 %r12, %r12, %s_PW;
                mov.f32 %f1, {{Zero}};
                mov.u32 %r13, %r2;
            WP:
                setp.ge.u32 %p1, %r13, %r10;
                @%p1 bra WP_END;
                rem.u32 %r14, %r13, %s_OW;
                div.u32 %r15, %r13, %s_OW;
                rem.u32 %r16, %r15, %s_OH;
                div.u32 %r17, %r15, %s_OH;
                mad.lo.u32 %r18, %r16, %s_SH, %r11;
                mad.lo.u32 %r19, %r14, %s_SW, %r12;
                setp.lt.u32 %p2, %r18, %s_H;
                setp.lt.and.u32 %p2, %r19, %s_W, %p2;
                @!%p2 bra WP_NEXT;
                mad.lo.u32 %r20, %r17, %s_F, %r7;
                mad.lo.u32 %r20, %r20, %s_OH, %r16;
                mad.lo.u32 %r20, %r20, %s_OW, %r14;
                mul.wide.u32 %rd4, %r20, 4;
                add.u64 %rd4, %rd2, %rd4;
                ld.global.f32 %f2, [%rd4];
                mad.lo.u32 %r21, %r17, %s_C, %r8;
                mad.lo.u32 %r21, %r21, %s_H, %r18;
                mad.lo.u32 %r21, %r21, %s_W, %r19;
                mul.wide.u32 %rd5, %r21, 4;
                add.u64 %rd5, %rd1, %rd5;
                ld.global.f32 %f3, [%rd5];
                fma.rn.f32 %f1, %f2, %f3, %f1;
            WP_NEXT:
                add.u32 %r13, %r13, %r3;
                bra WP;
            WP_END:
                mov.u32 %r22, red;
                shl.b32 %r23, %r2, 2;
                add.u32 %r23, %r22, %r23;
                st.shared.f32 [%r23], %f1;
                bar.sync 0;
                shr.u32 %r24, %r3, 1;
            TREE:
                setp.eq.u32 %p3, %r24, 0;
                @%p3 bra TREE_END;
                setp.lt.u32 %p4, %r2, %r24;
                @!%p4 bra TREE_WAIT;
                shl.b32 %r25, %r24, 2;
                add.u32 %r25, %r23, %r25;
                ld.shared.f32 %f4, [%r25];
                ld.shared.f32 %f5, [%r23];
                add.f32 %f5, %f5, %f4;
                st.shared.f32 [%r23], %f5;
            TREE_WAIT:
                bar.sync 0;
                shr.u32 %r24, %r24, 1;
                bra TREE;
            TREE_END:
                setp.ne.u32 %p5, %r2, 0;
                @%p5 bra DONE;
                ld.shared.f32 %f6, [%r22];
                mul.wide.u32 %rd6, %r1, 4;
                add.u64 %rd6, %rd3, %rd6;
                ld.global.f32 %f7, [%rd6];
                add.f32 %f7, %f7, %f6;
                st.global.f32 [%rd6], %f7;
            DONE:
                ret;
            }

            """);
        sb.Append(p);
    }

    // Resampling and per-channel normalization (Backend.ResizeNormalize): a thread per output takes the vertical taps in
    // order, each the horizontal pass of its row (the taps in order) computed in place: the two-pass result without the
    // intermediate image; floats, or Pillow's 8-bit passes in integers (2^21 + Σ byte · weight, >> 22, clipped).
    private static void ResizeNormalize(StringBuilder sb)
    {
        // Byte %r28 of the packed words into %r30.
        const string ReadByte = """
            shr.u32 %r31, %r28, 2;
            mul.wide.u32 %rd4, %r31, 4;
            add.u64 %rd4, %b_x, %rd4;
            ld.global.u32 %r30, [%rd4];
            and.b32 %r31, %r28, 3;
            shl.b32 %r31, %r31, 3;
            shr.u32 %r30, %r30, %r31;
            and.b32 %r30, %r30, 255;
            """;

        // The horizontal pass of row %r20 into %f10 (a float, or the byte as a float); without horizontal taps, the input.
        static string Row(string label) => $"""
            mul.lo.u32 %r21, %r20, %s_width;
            add.u32 %r21, %r21, %r9;
            setp.eq.u32 %p2, %s_xtaps, 0;
            @%p2 bra {label}_NONE;
            mul.lo.u32 %r22, %r5, %r10;
            mul.wide.u32 %rd1, %r22, 4;
            add.u64 %rd1, %b_coefficients, %rd1;
            ld.global.u32 %r23, [%rd1];
            ld.global.u32 %r24, [%rd1+4];
            add.u32 %r21, %r21, %r23;
            mov.f32 %f10, {Zero};
            mov.u32 %r25, 2097152;
            mov.u32 %r26, 0;
            {label}_TAP:
            setp.ge.u32 %p3, %r26, %r24;
            @%p3 bra {label}_SUMMED;
            add.u32 %r27, %r22, %r26;
            add.u32 %r27, %r27, 2;
            mul.wide.u32 %rd2, %r27, 4;
            add.u64 %rd2, %b_coefficients, %rd2;
            add.u32 %r28, %r21, %r26;
            @%p1 bra {label}_BYTE;
            ld.global.f32 %f11, [%rd2];
            mul.wide.u32 %rd3, %r28, 4;
            add.u64 %rd3, %b_x, %rd3;
            ld.global.f32 %f12, [%rd3];
            mul.rn.f32 %f12, %f12, %f11;
            add.rn.f32 %f10, %f10, %f12;
            bra {label}_NEXT;
            {label}_BYTE:
            ld.global.u32 %r29, [%rd2];
            {ReadByte}
            mad.lo.s32 %r25, %r30, %r29, %r25;
            {label}_NEXT:
            add.u32 %r26, %r26, 1;
            bra {label}_TAP;
            {label}_SUMMED:
            @!%p1 bra {label}_DONE;
            shr.s32 %r30, %r25, 22;
            max.s32 %r30, %r30, 0;
            min.s32 %r30, %r30, 255;
            cvt.rn.f32.s32 %f10, %r30;
            bra {label}_DONE;
            {label}_NONE:
            add.u32 %r28, %r21, %r5;
            @%p1 bra {label}_NONE_BYTE;
            mul.wide.u32 %rd3, %r28, 4;
            add.u64 %rd3, %b_x, %rd3;
            ld.global.f32 %f10, [%rd3];
            bra {label}_DONE;
            {label}_NONE_BYTE:
            {ReadByte}
            cvt.rn.f32.u32 %f10, %r30;
            {label}_DONE:
            """;

        Elementwise(sb, "resize_normalize_f32", ["x", "coefficients", "values", "y"],
            [("u32", "planes"), ("u32", "channels"), ("u32", "height"), ("u32", "width"), ("u32", "outheight"), ("u32", "outwidth"), ("u32", "xtaps"), ("u32", "ytaps"), ("u32", "bytes")],
            $"""
            rem.u32 %r5, %i, %s_outwidth;
            div.u32 %r6, %i, %s_outwidth;
            rem.u32 %r7, %r6, %s_outheight;
            div.u32 %r8, %r6, %s_outheight;
            mul.lo.u32 %r9, %s_height, %s_width;
            mul.lo.u32 %r9, %r9, %r8;
            add.u32 %r10, %s_xtaps, 2;
            setp.ne.u32 %p1, %s_bytes, 0;
            setp.eq.u32 %p4, %s_ytaps, 0;
            @%p4 bra DOWN_NONE;
            mul.lo.u32 %r11, %s_outwidth, %r10;
            setp.eq.u32 %p5, %s_xtaps, 0;
            selp.u32 %r11, 0, %r11, %p5;
            add.u32 %r12, %s_ytaps, 2;
            mad.lo.u32 %r11, %r7, %r12, %r11;
            mul.wide.u32 %rd5, %r11, 4;
            add.u64 %rd5, %b_coefficients, %rd5;
            ld.global.u32 %r13, [%rd5];
            ld.global.u32 %r14, [%rd5+4];
            mov.f32 %f1, {Zero};
            mov.u32 %r15, 2097152;
            mov.u32 %r16, 0;
            DOWN_TAP:
            setp.ge.u32 %p6, %r16, %r14;
            @%p6 bra DOWN_SUMMED;
            add.u32 %r20, %r13, %r16;
            {Row("A")}
            add.u32 %r17, %r11, %r16;
            add.u32 %r17, %r17, 2;
            mul.wide.u32 %rd6, %r17, 4;
            add.u64 %rd6, %b_coefficients, %rd6;
            @%p1 bra DOWN_BYTE;
            ld.global.f32 %f2, [%rd6];
            mul.rn.f32 %f2, %f10, %f2;
            add.rn.f32 %f1, %f1, %f2;
            bra DOWN_NEXT;
            DOWN_BYTE:
            ld.global.u32 %r18, [%rd6];
            cvt.rzi.s32.f32 %r19, %f10;
            mad.lo.s32 %r15, %r19, %r18, %r15;
            DOWN_NEXT:
            add.u32 %r16, %r16, 1;
            bra DOWN_TAP;
            DOWN_SUMMED:
            @!%p1 bra MAP;
            shr.s32 %r19, %r15, 22;
            max.s32 %r19, %r19, 0;
            min.s32 %r19, %r19, 255;
            cvt.rn.f32.s32 %f1, %r19;
            bra MAP;
            DOWN_NONE:
            mov.u32 %r20, %r7;
            {Row("B")}
            mov.f32 %f1, %f10;
            MAP:
            rem.u32 %r12, %r8, %s_channels;
            @%p1 bra MAP_BYTE;
            shl.b32 %r13, %r12, 1;
            mul.wide.u32 %rd7, %r13, 4;
            add.u64 %rd7, %b_values, %rd7;
            ld.global.f32 %f3, [%rd7];
            ld.global.f32 %f4, [%rd7+4];
            mul.rn.f32 %f1, %f1, %f3;
            add.rn.f32 %f1, %f1, %f4;
            bra STORE;
            MAP_BYTE:
            cvt.rzi.u32.f32 %r13, %f1;
            mad.lo.u32 %r13, %r12, 256, %r13;
            mul.wide.u32 %rd7, %r13, 4;
            add.u64 %rd7, %b_values, %rd7;
            ld.global.f32 %f1, [%rd7];
            STORE:
            st.global.f32 [%a_y], %f1;
            """);
    }

    // ------------------------------------------------------------------ the implicit products

    // One implicit product kernel (see the file's comment): blocked (4 × 4 outputs a thread, 64 × 64 tiles) or tiled (one
    // output a thread, 16 × 16 tiles); 256 threads, (tx, ty) = (tid % 16, tid / 16).
    private static void ConvProduct(StringBuilder sb, string name, ConvPass pass, bool blocked)
    {
        string[] pointers = pass switch
        {
            ConvPass.Forward => ["x", "w", "bias", "y"],
            ConvPass.Input => ["dy", "w", "dx"],
            _ => ["x", "dy", "dw", "part"],
        };
        int edge = blocked ? ConvBlockedEdge : ConvTileEdge, per = blocked ? 4 : 1, sharedLength = blocked ? ConvStep * ConvStride : ConvTileEdge * ConvTileEdge;
        var p = new StringBuilder();
        p.AppendLine($".visible .entry {name}(");
        p.AppendLine(string.Join(",\n", pointers.Select(q => $"    .param .u64 p_{q}").Concat(ConvScalars.Select(s => $"    .param .u32 p_{s}"))));
        p.AppendLine(")");
        p.AppendLine("{");
        p.AppendLine("    .reg .pred %p<16>;");
        p.AppendLine("    .reg .f32 %f<32>;");
        p.AppendLine("    .reg .f32 %acc<16>;");
        p.AppendLine("    .reg .b32 %r<64>;");
        p.AppendLine("    .reg .b64 %rd<16>;");
        p.AppendLine("    .reg .u32 " + string.Join(", ", ConvScalars.Select(s => $"%s_{s}")) + ";");
        p.AppendLine("    .reg .u32 %area, %ohow, %hw, %fg, %cg, %patch, %batch, %bi, %n, %q, %sp, %rows, %cols, %k0, %k1, %t, %me, %tx, %ty, %rowBase, %colBase;");
        p.AppendLine("    .reg .u32 %sA, %sB, %aRead, %bRead;");
        p.AppendLine("    .reg .u32 %lr<4>, %lka<4>, %lcol<4>, %lkb<4>, %lsa<4>, %lsb<4>;");
        p.AppendLine("    .reg .u64 " + string.Join(", ", pointers.Select(q => $"%g_{q}")) + ";");
        p.AppendLine($"    .shared .align 4 .f32 As[{sharedLength}];");
        p.AppendLine($"    .shared .align 4 .f32 Bs[{sharedLength}];");
        foreach (string s in ConvScalars)
        {
            p.AppendLine($"    ld.param.u32 %s_{s}, [p_{s}];");
        }

        foreach (string q in pointers)
        {
            p.AppendLine($"    ld.param.u64 %g_{q}, [p_{q}];");
            p.AppendLine($"    cvta.to.global.u64 %g_{q}, %g_{q};");
        }

        p.AppendLine($"""
                mul.lo.u32 %area, %s_KH, %s_KW;
                mul.lo.u32 %ohow, %s_OH, %s_OW;
                mul.lo.u32 %hw, %s_H, %s_W;
                div.u32 %fg, %s_F, %s_G;
                div.u32 %cg, %s_C, %s_G;
                mul.lo.u32 %patch, %cg, %area;
                mov.u32 %me, %tid.x;
                and.b32 %tx, %me, 15;
                shr.u32 %ty, %me, 4;
                mov.u32 %r1, %ctaid.y;
                mul.lo.u32 %rowBase, %r1, {edge};
                mov.u32 %r1, %ctaid.x;
                mul.lo.u32 %colBase, %r1, {edge};
                mov.u32 %sA, As;
                mov.u32 %sB, Bs;
            """);

        // Where each thread's staged elements come from and go: blocked, element l = tid + 256·j of each tile; tiled, (ty, tx).
        if (blocked)
        {
            for (int j = 0; j < 4; j++)
            {
                p.AppendLine($"""
                        add.u32 %r1, %me, {j * ConvThreads};
                        shr.u32 %lr{j}, %r1, 4;
                        and.b32 %lka{j}, %r1, 15;
                        and.b32 %lcol{j}, %r1, 63;
                        shr.u32 %lkb{j}, %r1, 6;
                        mad.lo.u32 %r2, %lka{j}, {ConvStride}, %lr{j};
                        shl.b32 %r2, %r2, 2;
                        add.u32 %lsa{j}, %sA, %r2;
                        mad.lo.u32 %r2, %lkb{j}, {ConvStride}, %lcol{j};
                        shl.b32 %r2, %r2, 2;
                        add.u32 %lsb{j}, %sB, %r2;
                    """);
            }

            p.AppendLine("""
                    shl.b32 %r2, %ty, 2;
                    add.u32 %aRead, %sA, %r2;
                    shl.b32 %r2, %tx, 2;
                    add.u32 %bRead, %sB, %r2;
                """);
        }
        else
        {
            p.AppendLine($"""
                    mov.u32 %lr0, %ty;
                    mov.u32 %lka0, %tx;
                    mov.u32 %lcol0, %tx;
                    mov.u32 %lkb0, %ty;
                    shl.b32 %r2, %me, 2;
                    add.u32 %lsa0, %sA, %r2;
                    add.u32 %lsb0, %sB, %r2;
                    mul.lo.u32 %r2, %ty, {ConvTileEdge * 4};
                    add.u32 %aRead, %sA, %r2;
                    shl.b32 %r2, %tx, 2;
                    add.u32 %bRead, %sB, %r2;
                """);
        }

        p.AppendLine(pass switch
        {
            ConvPass.Weight => "    mul.lo.u32 %batch, %s_G, %s_splits;",
            _ => "    mul.lo.u32 %batch, %s_N, %s_G;",
        });
        p.AppendLine("""
                mov.u32 %bi, %ctaid.z;
            BATCH:
                setp.ge.u32 %p1, %bi, %batch;
                @%p1 bra DONE;
            """);
        p.AppendLine(pass switch
        {
            ConvPass.Forward => """
                    div.u32 %n, %bi, %s_G;
                    rem.u32 %q, %bi, %s_G;
                    mov.u32 %rows, %fg;
                    mov.u32 %cols, %ohow;
                    mov.u32 %k0, 0;
                    mov.u32 %k1, %patch;
                """,
            ConvPass.Input => """
                    div.u32 %n, %bi, %s_G;
                    rem.u32 %q, %bi, %s_G;
                    mov.u32 %rows, %cg;
                    mov.u32 %cols, %hw;
                    mov.u32 %k0, 0;
                    mul.lo.u32 %k1, %fg, %area;
                """,
            _ => $"""
                    div.u32 %q, %bi, %s_splits;
                    rem.u32 %sp, %bi, %s_splits;
                    mov.u32 %rows, %fg;
                    mov.u32 %cols, %patch;
                    mul.lo.u32 %r1, %ohow, %s_N;
                    add.u32 %r2, %r1, %s_splits;
                    sub.u32 %r2, %r2, 1;
                    div.u32 %r2, %r2, %s_splits;
                    add.u32 %r2, %r2, {ConvStep - 1};
                    and.b32 %r2, %r2, 0xFFFFFFF0;
                    mul.lo.u32 %k0, %sp, %r2;
                    min.u32 %k0, %k0, %r1;
                    add.u32 %k1, %k0, %r2;
                    min.u32 %k1, %k1, %r1;
                """,
        });

        for (int i = 0; i < per * per; i++)
        {
            p.AppendLine($"    mov.f32 %acc{i}, {Zero};");
        }

        int step = blocked ? ConvStep : ConvTileEdge;
        p.AppendLine("""
                mov.u32 %t, %k0;
            STEP:
                setp.ge.u32 %p2, %t, %k1;
                @%p2 bra STORE;
            """);
        int loads = blocked ? 4 : 1;
        for (int j = 0; j < loads; j++)
        {
            p.AppendLine($"""
                    add.u32 %r40, %rowBase, %lr{j};
                    add.u32 %r41, %t, %lka{j};
                    mov.f32 %f1, {Zero};
                    setp.lt.u32 %p3, %r40, %rows;
                    setp.lt.and.u32 %p3, %r41, %k1, %p3;
                    @!%p3 bra A_SKIP_{j};
                """);
            p.AppendLine(LoadA(pass));
            p.AppendLine($"""
                A_SKIP_{j}:
                    st.shared.f32 [%lsa{j}], %f1;
                    add.u32 %r40, %t, %lkb{j};
                    add.u32 %r41, %colBase, %lcol{j};
                    mov.f32 %f1, {Zero};
                    setp.lt.u32 %p3, %r40, %k1;
                    setp.lt.and.u32 %p3, %r41, %cols, %p3;
                    @!%p3 bra B_SKIP_{j};
                """);
            p.AppendLine(LoadB(pass, $"B_SKIP_{j}"));
            p.AppendLine($"""
                B_SKIP_{j}:
                    st.shared.f32 [%lsb{j}], %f1;
                """);
        }

        p.AppendLine("    bar.sync 0;");
        for (int e = 0; e < step; e++)
        {
            if (blocked)
            {
                for (int i = 0; i < 4; i++)
                {
                    p.AppendLine($"    ld.shared.f32 %f{2 + i}, [%aRead+{4 * (e * ConvStride + 16 * i)}];");
                    p.AppendLine($"    ld.shared.f32 %f{6 + i}, [%bRead+{4 * (e * ConvStride + 16 * i)}];");
                }

                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        p.AppendLine($"    fma.rn.f32 %acc{i * 4 + j}, %f{2 + i}, %f{6 + j}, %acc{i * 4 + j};");
                    }
                }
            }
            else
            {
                p.AppendLine($"    ld.shared.f32 %f2, [%aRead+{4 * e}];");
                p.AppendLine($"    ld.shared.f32 %f3, [%bRead+{4 * e * ConvTileEdge}];");
                p.AppendLine("    fma.rn.f32 %acc0, %f2, %f3, %acc0;");
            }
        }

        p.AppendLine($"""
                bar.sync 0;
                add.u32 %t, %t, {step};
                bra STEP;
            STORE:
            """);
        for (int i = 0; i < per; i++)
        {
            for (int j = 0; j < per; j++)
            {
                p.AppendLine($"""
                        add.u32 %r40, %rowBase, %ty;
                        add.u32 %r40, %r40, {16 * i};
                        add.u32 %r41, %colBase, %tx;
                        add.u32 %r41, %r41, {16 * j};
                        setp.lt.u32 %p4, %r40, %rows;
                        setp.lt.and.u32 %p4, %r41, %cols, %p4;
                        @!%p4 bra S_SKIP_{i}_{j};
                        mov.f32 %f10, %acc{i * per + j};
                    """);
                p.AppendLine(Store(pass, $"S_{i}_{j}"));
                p.AppendLine($"S_SKIP_{i}_{j}:");
            }
        }

        p.AppendLine("""
                mov.u32 %r1, %nctaid.z;
                add.u32 %bi, %bi, %r1;
                bra BATCH;
            DONE:
                ret;
            }

            """);
        sb.Append(p);
    }

    // The left operand at row %r40, sum index %r41 (both in range) into %f1.
    private static string LoadA(ConvPass pass) => pass switch
    {
        // weight[(q·Fg + row)·patch + kk]
        ConvPass.Forward => """
                mad.lo.u32 %r42, %q, %fg, %r40;
                mad.lo.u32 %r42, %r42, %patch, %r41;
                mul.wide.u32 %rd1, %r42, 4;
                add.u64 %rd1, %g_w, %rd1;
                ld.global.f32 %f1, [%rd1];
            """,

        // weight[(q·Fg + f)·patch + c·area + r], kk = (f, r)
        ConvPass.Input => """
                div.u32 %r42, %r41, %area;
                rem.u32 %r43, %r41, %area;
                mad.lo.u32 %r44, %q, %fg, %r42;
                mul.lo.u32 %r44, %r44, %patch;
                mad.lo.u32 %r44, %r40, %area, %r44;
                add.u32 %r44, %r44, %r43;
                mul.wide.u32 %rd1, %r44, 4;
                add.u64 %rd1, %g_w, %rd1;
                ld.global.f32 %f1, [%rd1];
            """,

        // dy[(n·F + q·Fg + row)·OHOW + p], kk = (n, p)
        _ => """
                div.u32 %r42, %r41, %ohow;
                rem.u32 %r43, %r41, %ohow;
                mad.lo.u32 %r44, %q, %fg, %r40;
                mad.lo.u32 %r44, %r42, %s_F, %r44;
                mad.lo.u32 %r44, %r44, %ohow, %r43;
                mul.wide.u32 %rd1, %r44, 4;
                add.u64 %rd1, %g_dy, %rd1;
                ld.global.f32 %f1, [%rd1];
            """,
    };

    // The right operand at sum index %r40, column %r41 (both in range) into %f1, skipping to `skip` (leaving 0) outside the data.
    private static string LoadB(ConvPass pass, string skip) => pass switch
    {
        // x[n, q·Cg + c, oh·SH - PH + kh·DH, ow·SW - PW + kw·DW], kk = (c, kh, kw), column = (oh, ow)
        ConvPass.Forward => $"""
                div.u32 %r42, %r40, %area;
                rem.u32 %r43, %r40, %area;
                div.u32 %r44, %r43, %s_KW;
                rem.u32 %r45, %r43, %s_KW;
                div.u32 %r46, %r41, %s_OW;
                rem.u32 %r47, %r41, %s_OW;
                mul.lo.u32 %r48, %r44, %s_DH;
                mad.lo.u32 %r48, %r46, %s_SH, %r48;
                sub.u32 %r48, %r48, %s_PH;
                mul.lo.u32 %r49, %r45, %s_DW;
                mad.lo.u32 %r49, %r47, %s_SW, %r49;
                sub.u32 %r49, %r49, %s_PW;
                setp.lt.u32 %p5, %r48, %s_H;
                setp.lt.and.u32 %p5, %r49, %s_W, %p5;
                @!%p5 bra {skip};
                mad.lo.u32 %r50, %q, %cg, %r42;
                mad.lo.u32 %r50, %n, %s_C, %r50;
                mad.lo.u32 %r50, %r50, %s_H, %r48;
                mad.lo.u32 %r50, %r50, %s_W, %r49;
                mul.wide.u32 %rd2, %r50, 4;
                add.u64 %rd2, %g_x, %rd2;
                ld.global.f32 %f1, [%rd2];
            """,

        // dy[n, q·Fg + f, oh, ow] with oh·SH = ih + PH - kh·DH, ow·SW = iw + PW - kw·DW; kk = (f, kh, kw), column = (ih, iw)
        ConvPass.Input => $"""
                div.u32 %r42, %r40, %area;
                rem.u32 %r43, %r40, %area;
                div.u32 %r44, %r43, %s_KW;
                rem.u32 %r45, %r43, %s_KW;
                div.u32 %r46, %r41, %s_W;
                rem.u32 %r47, %r41, %s_W;
                add.u32 %r48, %r46, %s_PH;
                mul.lo.u32 %r51, %r44, %s_DH;
                sub.u32 %r48, %r48, %r51;
                setp.lt.s32 %p5, %r48, 0;
                @%p5 bra {skip};
                add.u32 %r49, %r47, %s_PW;
                mul.lo.u32 %r51, %r45, %s_DW;
                sub.u32 %r49, %r49, %r51;
                setp.lt.s32 %p5, %r49, 0;
                @%p5 bra {skip};
                rem.u32 %r51, %r48, %s_SH;
                rem.u32 %r52, %r49, %s_SW;
                or.b32 %r51, %r51, %r52;
                setp.ne.u32 %p5, %r51, 0;
                @%p5 bra {skip};
                div.u32 %r48, %r48, %s_SH;
                div.u32 %r49, %r49, %s_SW;
                setp.ge.u32 %p5, %r48, %s_OH;
                @%p5 bra {skip};
                setp.ge.u32 %p5, %r49, %s_OW;
                @%p5 bra {skip};
                mad.lo.u32 %r50, %q, %fg, %r42;
                mad.lo.u32 %r50, %n, %s_F, %r50;
                mad.lo.u32 %r50, %r50, %s_OH, %r48;
                mad.lo.u32 %r50, %r50, %s_OW, %r49;
                mul.wide.u32 %rd2, %r50, 4;
                add.u64 %rd2, %g_dy, %rd2;
                ld.global.f32 %f1, [%rd2];
            """,

        // x[n, q·Cg + c, ih, iw]; kk = (n, oh, ow), column = (c, kh, kw)
        _ => $"""
                div.u32 %r42, %r40, %ohow;
                rem.u32 %r43, %r40, %ohow;
                div.u32 %r46, %r43, %s_OW;
                rem.u32 %r47, %r43, %s_OW;
                div.u32 %r52, %r41, %area;
                rem.u32 %r43, %r41, %area;
                div.u32 %r44, %r43, %s_KW;
                rem.u32 %r45, %r43, %s_KW;
                mul.lo.u32 %r48, %r44, %s_DH;
                mad.lo.u32 %r48, %r46, %s_SH, %r48;
                sub.u32 %r48, %r48, %s_PH;
                mul.lo.u32 %r49, %r45, %s_DW;
                mad.lo.u32 %r49, %r47, %s_SW, %r49;
                sub.u32 %r49, %r49, %s_PW;
                setp.lt.u32 %p5, %r48, %s_H;
                setp.lt.and.u32 %p5, %r49, %s_W, %p5;
                @!%p5 bra {skip};
                mad.lo.u32 %r50, %q, %cg, %r52;
                mad.lo.u32 %r50, %r42, %s_C, %r50;
                mad.lo.u32 %r50, %r50, %s_H, %r48;
                mad.lo.u32 %r50, %r50, %s_W, %r49;
                mul.wide.u32 %rd2, %r50, 4;
                add.u64 %rd2, %g_x, %rd2;
                ld.global.f32 %f1, [%rd2];
            """,
    };

    // Stores the sum %f10 for row %r40, column %r41.
    private static string Store(ConvPass pass, string label) => pass switch
    {
        // y[(n·F + f)·OHOW + column] = act(sum + bias[f]), f = q·Fg + row
        ConvPass.Forward => $"""
                mad.lo.u32 %r53, %q, %fg, %r40;
                and.b32 %r54, %s_flags, 1;
                setp.eq.u32 %p6, %r54, 0;
                @%p6 bra {label}_ACT;
                mul.wide.u32 %rd3, %r53, 4;
                add.u64 %rd3, %g_bias, %rd3;
                ld.global.f32 %f11, [%rd3];
                add.f32 %f10, %f10, %f11;
            {label}_ACT:
                shr.u32 %r54, %s_flags, 1;
                {Activation("%f10", "%r54", label)}
                mad.lo.u32 %r55, %n, %s_F, %r53;
                mad.lo.u32 %r55, %r55, %ohow, %r41;
                mul.wide.u32 %rd3, %r55, 4;
                add.u64 %rd3, %g_y, %rd3;
                st.global.f32 [%rd3], %f10;
            """,

        // dx[(n·C + q·Cg + row)·HW + column] += sum
        ConvPass.Input => """
                mad.lo.u32 %r53, %q, %cg, %r40;
                mad.lo.u32 %r53, %n, %s_C, %r53;
                mad.lo.u32 %r53, %r53, %hw, %r41;
                mul.wide.u32 %rd3, %r53, 4;
                add.u64 %rd3, %g_dx, %rd3;
                ld.global.f32 %f11, [%rd3];
                add.f32 %f11, %f11, %f10;
                st.global.f32 [%rd3], %f11;
            """,

        // dw[(q·Fg + row)·patch + column] += sum (one split), else part[sp·F·patch + that] = sum
        _ => $"""
                mad.lo.u32 %r53, %q, %fg, %r40;
                mad.lo.u32 %r53, %r53, %patch, %r41;
                setp.ne.u32 %p6, %s_splits, 1;
                @%p6 bra {label}_PART;
                mul.wide.u32 %rd3, %r53, 4;
                add.u64 %rd3, %g_dw, %rd3;
                ld.global.f32 %f11, [%rd3];
                add.f32 %f11, %f11, %f10;
                st.global.f32 [%rd3], %f11;
                bra {label}_END;
            {label}_PART:
                mul.lo.u32 %r54, %s_F, %patch;
                mad.lo.u32 %r54, %sp, %r54, %r53;
                mul.wide.u32 %rd3, %r54, 4;
                add.u64 %rd3, %g_part, %rd3;
                st.global.f32 [%rd3], %f10;
            {label}_END:
            """,
    };
}
