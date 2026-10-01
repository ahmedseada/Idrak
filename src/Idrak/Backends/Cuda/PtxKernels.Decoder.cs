// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;

namespace Idrak.Backends.Cuda;

// PTX for decoder-only language model layers: RMS normalization and rotary position embeddings.
internal static partial class PtxKernels
{
    public static readonly string[] DecoderNames = ["rms_norm_f32", "rms_norm_backward_f32", "rope_f32", "rms_norm_affine_f32", "gated_act_f32", "gated_act_bwd_f32", "gated_act_bf16_f32", "gated_act_bwd_bf16_f32", "add_rms_norm_affine_f32", "rms_norm_rope_f32", "rms_norm_rope2_f32", "softmax_ce_rows_f32", "norm_rope_heads_f32"];

    private static void BuildDecoder(StringBuilder sb)
    {
        const string RowStart = """
            mul.lo.u32 %r5, %row, %s_cols;
            mul.wide.u32 %rd1, %r5, 4;
            """;

        // y = x · inv, inv = 1 / sqrt(mean(x²) + eps) per row (stored for the backward pass): one block per row.
        RowBlock(sb, "rms_norm_f32", ["x", "y", "inv"], [("u32", "cols"), ("f32", "eps")],
            RowStart + $"""
            add.u64 %rd2, %b_x, %rd1;
            add.u64 %rd3, %b_y, %rd1;
            sub.u64 %rd6, %rd3, %rd2;
            cvt.rn.f32.u32 %f10, %s_cols;
            mov.f32 %f1, {Zero};
            """ + "\n" + StridedLoop("RS", "%rd2", "%s_cols", "fma.rn.f32 %f1, %f2, %f2, %f1;") + "\n"
            + BlockReduce("RSUM", "%f1", "add", Zero) + "\n" + """
            div.rn.f32 %f1, %f1, %f10;
            add.f32 %f1, %f1, %s_eps;
            sqrt.rn.f32 %f1, %f1;
            rcp.rn.f32 %f1, %f1;
            setp.eq.u32 %p5, %tx, 0;
            @%p5 st.global.f32 [%a_inv], %f1;
            """ + "\n" + StridedLoop("RW", "%rd2", "%s_cols", """
            mul.f32 %f2, %f2, %f1;
            add.u64 %rd7, %rd5, %rd6;
            st.global.f32 [%rd7], %f2;
            """));

        // dx += inv · (dy - y · mean(dy · y)) per row (y is the normalized output): one block per row.
        RowBlock(sb, "rms_norm_backward_f32", ["dy", "y", "inv", "dx"], [("u32", "cols")],
            RowStart + $"""
            add.u64 %rd2, %b_dy, %rd1;
            add.u64 %rd3, %b_y, %rd1;
            sub.u64 %rd6, %rd3, %rd2;
            add.u64 %rd4, %b_dx, %rd1;
            sub.u64 %rd8, %rd4, %rd2;
            cvt.rn.f32.u32 %f10, %s_cols;
            mov.f32 %f1, {Zero};
            """ + "\n" + StridedLoop("RD", "%rd2", "%s_cols", """
            add.u64 %rd7, %rd5, %rd6;
            ld.global.f32 %f3, [%rd7];
            fma.rn.f32 %f1, %f2, %f3, %f1;
            """) + "\n" + BlockReduce("RDOT", "%f1", "add", Zero) + "\n" + """
            div.rn.f32 %f1, %f1, %f10;
            ld.global.f32 %f6, [%a_inv];
            """ + "\n" + StridedLoop("RB", "%rd2", "%s_cols", """
            add.u64 %rd7, %rd5, %rd6;
            ld.global.f32 %f3, [%rd7];
            mul.f32 %f4, %f3, %f1;
            sub.f32 %f4, %f2, %f4;
            mul.f32 %f4, %f4, %f6;
            add.u64 %rd9, %rd5, %rd8;
            ld.global.f32 %f5, [%rd9];
            add.f32 %f5, %f5, %f4;
            st.global.f32 [%rd9], %f5;
            """));

        // y = x · inv · (gain + offset): normalization and gain in one pass (inference), one block per row.
        // Token cross-entropy per row (language-model training): loss[r] = w_r · (logsumexp(x_r) - x_r[t_r]), and x_r is
        // replaced by its gradient scale · w_r · (softmax(x_r) - onehot(t_r)). One block per row; every thread reads the
        // target logit before the block reductions (which synchronize) and the rewrite after them.
        RowBlock(sb, "softmax_ce_rows_f32", ["x", "targets", "weights", "losses"], [("u32", "cols"), ("f32", "scale")],
            RowStart + $"""
            add.u64 %rd2, %b_x, %rd1;
            ld.global.f32 %f20, [%a_weights];
            ld.global.f32 %f21, [%a_targets];
            cvt.rzi.u32.f32 %r20, %f21;
            mul.wide.u32 %rd8, %r20, 4;
            add.u64 %rd8, %rd8, %rd2;
            ld.global.f32 %f22, [%rd8];
            mov.f32 %f1, {NegInf};
            """ + "\n" + StridedLoop("CM", "%rd2", "%s_cols", "max.f32 %f1, %f1, %f2;") + "\n"
            + BlockReduce("CMAX", "%f1", "max", NegInf) + "\n" + $"""
            mov.f32 %f3, {Zero};
            """ + "\n" + StridedLoop("CS", "%rd2", "%s_cols", $"""
            sub.f32 %f4, %f2, %f1;
            mul.f32 %f4, %f4, {Log2E};
            ex2.approx.ftz.f32 %f4, %f4;
            add.f32 %f3, %f3, %f4;
            """) + "\n" + BlockReduce("CSUM", "%f3", "add", Zero) + "\n" + $"""
            lg2.approx.f32 %f5, %f3;
            fma.rn.f32 %f5, %f5, {Ln2}, %f1;
            setp.eq.u32 %p9, %tx, 0;
            sub.f32 %f7, %f5, %f22;
            mul.f32 %f7, %f7, %f20;
            @%p9 st.global.f32 [%a_losses], %f7;
            mul.f32 %f6, %f20, %s_scale;
            """ + "\n" + StridedLoop("CW", "%rd2", "%s_cols", $"""
            sub.f32 %f4, %f2, %f5;
            mul.f32 %f4, %f4, {Log2E};
            ex2.approx.ftz.f32 %f4, %f4;
            setp.eq.u32 %p10, %r6, %r20;
            @%p10 sub.f32 %f4, %f4, 0f3F800000;
            mul.f32 %f4, %f4, %f6;
            st.global.f32 [%rd5], %f4;
            """));

        RowBlock(sb, "rms_norm_affine_f32", ["x", "gain", "y"], [("u32", "cols"), ("f32", "eps"), ("f32", "offset")],
            RowStart + $"""
            add.u64 %rd2, %b_x, %rd1;
            add.u64 %rd3, %b_y, %rd1;
            sub.u64 %rd6, %rd3, %rd2;
            cvt.rn.f32.u32 %f10, %s_cols;
            mov.f32 %f1, {Zero};
            """ + "\n" + StridedLoop("RS", "%rd2", "%s_cols", "fma.rn.f32 %f1, %f2, %f2, %f1;") + "\n"
            + BlockReduce("RSUM", "%f1", "add", Zero) + "\n" + """
            div.rn.f32 %f1, %f1, %f10;
            add.f32 %f1, %f1, %s_eps;
            sqrt.rn.f32 %f1, %f1;
            rcp.rn.f32 %f1, %f1;
            """ + "\n" + StridedLoop("RW", "%rd2", "%s_cols", """
            mul.wide.u32 %rd8, %r6, 4;
            add.u64 %rd8, %rd8, %b_gain;
            ld.global.f32 %f3, [%rd8];
            add.f32 %f3, %f3, %s_offset;
            mul.f32 %f2, %f2, %f1;
            mul.f32 %f2, %f2, %f3;
            add.u64 %rd7, %rd5, %rd6;
            st.global.f32 [%rd7], %f2;
            """));

        // sum = a + b (the residual addition, kept), y = sum · inv · (gain + offset): one block per row.
        RowBlock(sb, "add_rms_norm_affine_f32", ["a", "b", "sum", "gain", "y"], [("u32", "cols"), ("f32", "eps"), ("f32", "offset")],
            RowStart + $"""
            add.u64 %rd2, %b_a, %rd1;
            add.u64 %rd3, %b_b, %rd1;
            add.u64 %rd4, %b_sum, %rd1;
            add.u64 %rd9, %b_y, %rd1;
            sub.u64 %rd6, %rd3, %rd2;
            sub.u64 %rd10, %rd4, %rd2;
            sub.u64 %rd11, %rd9, %rd2;
            cvt.rn.f32.u32 %f10, %s_cols;
            mov.f32 %f1, {Zero};
            """ + "\n" + StridedLoop("AS", "%rd2", "%s_cols", """
            add.u64 %rd7, %rd5, %rd6;
            ld.global.f32 %f3, [%rd7];
            add.f32 %f2, %f2, %f3;
            add.u64 %rd7, %rd5, %rd10;
            st.global.f32 [%rd7], %f2;
            fma.rn.f32 %f1, %f2, %f2, %f1;
            """) + "\n" + BlockReduce("RSUM", "%f1", "add", Zero) + "\n" + """
            div.rn.f32 %f1, %f1, %f10;
            add.f32 %f1, %f1, %s_eps;
            sqrt.rn.f32 %f1, %f1;
            rcp.rn.f32 %f1, %f1;
            """ + "\n" + StridedLoop("AW", "%rd4", "%s_cols", """
            mul.wide.u32 %rd8, %r6, 4;
            add.u64 %rd8, %rd8, %b_gain;
            ld.global.f32 %f3, [%rd8];
            add.f32 %f3, %f3, %s_offset;
            mul.f32 %f2, %f2, %f1;
            mul.f32 %f2, %f2, %f3;
            sub.u64 %rd7, %rd5, %rd4;
            add.u64 %rd7, %rd7, %rd9;
            st.global.f32 [%rd7], %f2;
            """));

        // RMS normalization with gain of each head's vector, then the rotary embedding (as rope_f32) in one pass: rows
        // are batch·steps·heads vectors of cols values; the row's position is positions[(row / heads) % steps].
        RowBlock(sb, "rms_norm_rope_f32", ["x", "gain", "cos", "sin", "positions", "y"],
            [("u32", "cols"), ("f32", "eps"), ("f32", "offset"), ("u32", "heads"), ("u32", "steps"), ("u32", "half"), ("u32", "interleaved")],
            RowStart + $"""
            add.u64 %rd2, %b_x, %rd1;
            add.u64 %rd3, %b_y, %rd1;
            cvt.rn.f32.u32 %f10, %s_cols;
            mov.f32 %f1, {Zero};
            """ + "\n" + StridedLoop("RS", "%rd2", "%s_cols", "fma.rn.f32 %f1, %f2, %f2, %f1;") + "\n"
            + BlockReduce("RSUM", "%f1", "add", Zero) + "\n" + """
            div.rn.f32 %f1, %f1, %f10;
            add.f32 %f1, %f1, %s_eps;
            sqrt.rn.f32 %f1, %f1;
            rcp.rn.f32 %f1, %f1;
            div.u32 %r7, %row, %s_heads;
            rem.u32 %r7, %r7, %s_steps;
            mul.wide.u32 %rd4, %r7, 4;
            add.u64 %rd4, %rd4, %b_positions;
            ld.global.f32 %f4, [%rd4];
            cvt.rzi.u32.f32 %r8, %f4;
            mul.lo.u32 %r8, %r8, %s_half;
            setp.ne.u32 %p6, %s_interleaved, 0;
            mov.u32 %r9, %tx;
            RP:
            setp.ge.u32 %p7, %r9, %s_half;
            @%p7 bra RP_END;
            shl.b32 %r10, %r9, 1;
            add.u32 %r11, %r9, %s_half;
            add.u32 %r12, %r10, 1;
            selp.b32 %r13, %r10, %r9, %p6;
            selp.b32 %r14, %r12, %r11, %p6;
            mul.wide.u32 %rd5, %r13, 4;
            mul.wide.u32 %rd6, %r14, 4;
            add.u64 %rd7, %rd5, %rd2;
            ld.global.f32 %f2, [%rd7];
            add.u64 %rd7, %rd6, %rd2;
            ld.global.f32 %f3, [%rd7];
            add.u64 %rd7, %rd5, %b_gain;
            ld.global.f32 %f5, [%rd7];
            add.f32 %f5, %f5, %s_offset;
            add.u64 %rd7, %rd6, %b_gain;
            ld.global.f32 %f6, [%rd7];
            add.f32 %f6, %f6, %s_offset;
            mul.f32 %f2, %f2, %f1;
            mul.f32 %f2, %f2, %f5;
            mul.f32 %f3, %f3, %f1;
            mul.f32 %f3, %f3, %f6;
            add.u32 %r15, %r8, %r9;
            mul.wide.u32 %rd8, %r15, 4;
            add.u64 %rd9, %rd8, %b_cos;
            ld.global.f32 %f7, [%rd9];
            add.u64 %rd9, %rd8, %b_sin;
            ld.global.f32 %f8, [%rd9];
            mul.f32 %f9, %f3, %f8;
            neg.f32 %f9, %f9;
            fma.rn.f32 %f11, %f2, %f7, %f9;
            mul.f32 %f12, %f2, %f8;
            fma.rn.f32 %f13, %f3, %f7, %f12;
            add.u64 %rd7, %rd5, %rd3;
            st.global.f32 [%rd7], %f11;
            add.u64 %rd7, %rd6, %rd3;
            st.global.f32 [%rd7], %f13;
            add.u32 %r9, %r9, %nt;
            bra RP;
            RP_END:
            shl.b32 %r9, %s_half, 1;
            add.u32 %r9, %r9, %tx;
            PT:
            setp.ge.u32 %p7, %r9, %s_cols;
            @%p7 bra PT_END;
            mul.wide.u32 %rd5, %r9, 4;
            add.u64 %rd7, %rd5, %rd2;
            ld.global.f32 %f2, [%rd7];
            add.u64 %rd7, %rd5, %b_gain;
            ld.global.f32 %f5, [%rd7];
            add.f32 %f5, %f5, %s_offset;
            mul.f32 %f2, %f2, %f1;
            mul.f32 %f2, %f2, %f5;
            add.u64 %rd7, %rd5, %rd3;
            st.global.f32 [%rd7], %f2;
            add.u32 %r9, %r9, %nt;
            bra PT;
            PT_END:
            """);

        // The same for two tensors in one launch (the query and key heads): rows past rows1 belong to x2 (gain2, y2, heads2).
        RowBlock(sb, "rms_norm_rope2_f32", ["x", "gain", "cos", "sin", "positions", "y", "x2", "gain2", "y2"],
            [("u32", "cols"), ("f32", "eps"), ("f32", "offset"), ("u32", "heads"), ("u32", "steps"), ("u32", "half"), ("u32", "interleaved"),
                ("u32", "rows1"), ("f32", "eps2"), ("f32", "offset2"), ("u32", "heads2")],
            """
            setp.ge.u32 %p15, %row, %s_rows1;
            @%p15 sub.u32 %row, %row, %s_rows1;
            @%p15 mov.u64 %b_x, %b_x2;
            @%p15 mov.u64 %b_gain, %b_gain2;
            @%p15 mov.u64 %b_y, %b_y2;
            @%p15 mov.u32 %s_heads, %s_heads2;
            @%p15 mov.f32 %s_eps, %s_eps2;
            @%p15 mov.f32 %s_offset, %s_offset2;
            """ + RowStart + $"""
            add.u64 %rd2, %b_x, %rd1;
            add.u64 %rd3, %b_y, %rd1;
            cvt.rn.f32.u32 %f10, %s_cols;
            mov.f32 %f1, {Zero};
            """ + "\n" + StridedLoop("RS", "%rd2", "%s_cols", "fma.rn.f32 %f1, %f2, %f2, %f1;") + "\n"
            + BlockReduce("RSUM", "%f1", "add", Zero) + "\n" + """
            div.rn.f32 %f1, %f1, %f10;
            add.f32 %f1, %f1, %s_eps;
            sqrt.rn.f32 %f1, %f1;
            rcp.rn.f32 %f1, %f1;
            div.u32 %r7, %row, %s_heads;
            rem.u32 %r7, %r7, %s_steps;
            mul.wide.u32 %rd4, %r7, 4;
            add.u64 %rd4, %rd4, %b_positions;
            ld.global.f32 %f4, [%rd4];
            cvt.rzi.u32.f32 %r8, %f4;
            mul.lo.u32 %r8, %r8, %s_half;
            setp.ne.u32 %p6, %s_interleaved, 0;
            mov.u32 %r9, %tx;
            RP:
            setp.ge.u32 %p7, %r9, %s_half;
            @%p7 bra RP_END;
            shl.b32 %r10, %r9, 1;
            add.u32 %r11, %r9, %s_half;
            add.u32 %r12, %r10, 1;
            selp.b32 %r13, %r10, %r9, %p6;
            selp.b32 %r14, %r12, %r11, %p6;
            mul.wide.u32 %rd5, %r13, 4;
            mul.wide.u32 %rd6, %r14, 4;
            add.u64 %rd7, %rd5, %rd2;
            ld.global.f32 %f2, [%rd7];
            add.u64 %rd7, %rd6, %rd2;
            ld.global.f32 %f3, [%rd7];
            add.u64 %rd7, %rd5, %b_gain;
            ld.global.f32 %f5, [%rd7];
            add.f32 %f5, %f5, %s_offset;
            add.u64 %rd7, %rd6, %b_gain;
            ld.global.f32 %f6, [%rd7];
            add.f32 %f6, %f6, %s_offset;
            mul.f32 %f2, %f2, %f1;
            mul.f32 %f2, %f2, %f5;
            mul.f32 %f3, %f3, %f1;
            mul.f32 %f3, %f3, %f6;
            add.u32 %r15, %r8, %r9;
            mul.wide.u32 %rd8, %r15, 4;
            add.u64 %rd9, %rd8, %b_cos;
            ld.global.f32 %f7, [%rd9];
            add.u64 %rd9, %rd8, %b_sin;
            ld.global.f32 %f8, [%rd9];
            mul.f32 %f9, %f3, %f8;
            neg.f32 %f9, %f9;
            fma.rn.f32 %f11, %f2, %f7, %f9;
            mul.f32 %f12, %f2, %f8;
            fma.rn.f32 %f13, %f3, %f7, %f12;
            add.u64 %rd7, %rd5, %rd3;
            st.global.f32 [%rd7], %f11;
            add.u64 %rd7, %rd6, %rd3;
            st.global.f32 [%rd7], %f13;
            add.u32 %r9, %r9, %nt;
            bra RP;
            RP_END:
            shl.b32 %r9, %s_half, 1;
            add.u32 %r9, %r9, %tx;
            PT:
            setp.ge.u32 %p7, %r9, %s_cols;
            @%p7 bra PT_END;
            mul.wide.u32 %rd5, %r9, 4;
            add.u64 %rd7, %rd5, %rd2;
            ld.global.f32 %f2, [%rd7];
            add.u64 %rd7, %rd5, %b_gain;
            ld.global.f32 %f5, [%rd7];
            add.f32 %f5, %f5, %s_offset;
            mul.f32 %f2, %f2, %f1;
            mul.f32 %f2, %f2, %f5;
            add.u64 %rd7, %rd5, %rd3;
            st.global.f32 [%rd7], %f2;
            add.u32 %r9, %r9, %nt;
            bra PT;
            PT_END:
            """);

        // Queries, keys and values [batch, steps, heads·cols] (the projections) → the layouts attention reads, in one
        // launch: rows1 query rows, then rows2 key rows, then rows2 value rows (row = (b·steps + s)·heads + h). Query and
        // key heads are RMS-normalized with their gain when flags & 1, and rotated (half pairs, as rope_f32; half 0: no
        // rotation); values are copied. Queries go to y [batch, heads, steps, cols]; keys and values to y2 / y3
        // [batch, heads2, cap, stride] at row offset + s, offset = pos[0] when flags & 4 (a cache) else 0, as floats or,
        // when flags & 2, as bfloat16 (rounded to nearest even, as kv_write_bf16).
        const string Store = """
            add.u32 %r27, %r21, {1};
            @%p12 bra {2}_H;
            mul.wide.u32 %rd11, %r27, 4;
            add.u64 %rd11, %rd11, %rd10;
            st.global.f32 [%rd11], {0};
            bra {2}_D;
            {2}_H:
            mul.wide.u32 %rd11, %r27, 2;
            add.u64 %rd11, %rd11, %rd10;
            mov.b32 %r28, {0};
            shr.u32 %r29, %r28, 16;
            and.b32 %r29, %r29, 1;
            add.u32 %r29, %r29, 0x7FFF;
            add.u32 %r28, %r28, %r29;
            shr.u32 %r28, %r28, 16;
            st.global.u16 [%rd11], %r28;
            {2}_D:
            """;
        RowBlock(sb, "norm_rope_heads_f32", ["x", "gain", "x2", "gain2", "x3", "cos", "sin", "positions", "pos", "y", "y2", "y3"],
            [("u32", "cols"), ("f32", "eps"), ("f32", "offset"), ("u32", "heads"), ("f32", "eps2"), ("f32", "offset2"), ("u32", "heads2"),
                ("u32", "steps"), ("u32", "half"), ("u32", "interleaved"), ("u32", "rows1"), ("u32", "rows2"), ("u32", "flags"), ("u32", "cap"),
                ("u32", "stride")],
            $"""
            mov.u32 %r20, 0;
            mov.u64 %rd10, %b_y;
            mov.u32 %r24, %s_steps;
            mov.u32 %r25, 0;
            mov.u32 %r26, %s_cols;
            setp.ge.u32 %p15, %row, %s_rows1;
            @%p15 sub.u32 %row, %row, %s_rows1;
            @%p15 mov.u64 %b_x, %b_x2;
            @%p15 mov.u64 %b_gain, %b_gain2;
            @%p15 mov.u32 %s_heads, %s_heads2;
            @%p15 mov.f32 %s_eps, %s_eps2;
            @%p15 mov.f32 %s_offset, %s_offset2;
            @%p15 mov.u32 %r20, 1;
            @%p15 mov.u64 %rd10, %b_y2;
            @%p15 mov.u32 %r24, %s_cap;
            @%p15 mov.u32 %r26, %s_stride;
            setp.ge.and.u32 %p14, %row, %s_rows2, %p15;
            @%p14 sub.u32 %row, %row, %s_rows2;
            @%p14 mov.u64 %b_x, %b_x3;
            @%p14 mov.u32 %r20, 2;
            @%p14 mov.u64 %rd10, %b_y3;
            and.b32 %r16, %s_flags, 4;
            setp.ne.and.u32 %p13, %r16, 0, %p15;
            @%p13 ld.global.f32 %f14, [%b_pos];
            @%p13 cvt.rzi.u32.f32 %r25, %f14;
            and.b32 %r16, %s_flags, 2;
            setp.ne.and.u32 %p12, %r16, 0, %p15;
            and.b32 %r16, %s_flags, 1;
            setp.ne.u32 %p11, %r16, 0;
            setp.lt.and.u32 %p11, %r20, 2, %p11;
            selp.b32 %r23, 0, %s_half, %p14;
            rem.u32 %r16, %row, %s_heads;
            div.u32 %r17, %row, %s_heads;
            rem.u32 %r18, %r17, %s_steps;
            div.u32 %r17, %r17, %s_steps;
            mad.lo.u32 %r17, %r17, %s_heads, %r16;
            mad.lo.u32 %r17, %r17, %r24, %r25;
            add.u32 %r17, %r17, %r18;
            mul.lo.u32 %r21, %r17, %r26;
            """ + RowStart + $"""
            add.u64 %rd2, %b_x, %rd1;
            mov.f32 %f1, {One};
            @!%p11 bra NORM_END;
            cvt.rn.f32.u32 %f10, %s_cols;
            mov.f32 %f1, {Zero};
            """ + "\n" + StridedLoop("RS", "%rd2", "%s_cols", "fma.rn.f32 %f1, %f2, %f2, %f1;") + "\n"
            + BlockReduce("RSUM", "%f1", "add", Zero) + "\n" + """
            div.rn.f32 %f1, %f1, %f10;
            add.f32 %f1, %f1, %s_eps;
            sqrt.rn.f32 %f1, %f1;
            rcp.rn.f32 %f1, %f1;
            NORM_END:
            setp.eq.u32 %p10, %r23, 0;
            @%p10 bra RP_END;
            mul.wide.u32 %rd4, %r18, 4;
            add.u64 %rd4, %rd4, %b_positions;
            ld.global.f32 %f4, [%rd4];
            cvt.rzi.u32.f32 %r8, %f4;
            mul.lo.u32 %r8, %r8, %s_half;
            setp.ne.u32 %p6, %s_interleaved, 0;
            mov.u32 %r9, %tx;
            RP:
            setp.ge.u32 %p7, %r9, %r23;
            @%p7 bra RP_END;
            shl.b32 %r10, %r9, 1;
            add.u32 %r11, %r9, %r23;
            add.u32 %r12, %r10, 1;
            selp.b32 %r13, %r10, %r9, %p6;
            selp.b32 %r14, %r12, %r11, %p6;
            mul.wide.u32 %rd5, %r13, 4;
            mul.wide.u32 %rd6, %r14, 4;
            add.u64 %rd7, %rd5, %rd2;
            ld.global.f32 %f2, [%rd7];
            add.u64 %rd7, %rd6, %rd2;
            ld.global.f32 %f3, [%rd7];
            mov.f32 %f5, 0f3F800000;
            mov.f32 %f6, 0f3F800000;
            add.u64 %rd7, %rd5, %b_gain;
            @%p11 ld.global.f32 %f5, [%rd7];
            @%p11 add.f32 %f5, %f5, %s_offset;
            add.u64 %rd7, %rd6, %b_gain;
            @%p11 ld.global.f32 %f6, [%rd7];
            @%p11 add.f32 %f6, %f6, %s_offset;
            mul.f32 %f2, %f2, %f1;
            mul.f32 %f2, %f2, %f5;
            mul.f32 %f3, %f3, %f1;
            mul.f32 %f3, %f3, %f6;
            add.u32 %r15, %r8, %r9;
            mul.wide.u32 %rd8, %r15, 4;
            add.u64 %rd9, %rd8, %b_cos;
            ld.global.f32 %f7, [%rd9];
            add.u64 %rd9, %rd8, %b_sin;
            ld.global.f32 %f8, [%rd9];
            mul.f32 %f9, %f3, %f8;
            neg.f32 %f9, %f9;
            fma.rn.f32 %f11, %f2, %f7, %f9;
            mul.f32 %f12, %f2, %f8;
            fma.rn.f32 %f13, %f3, %f7, %f12;
            """ + "\n" + string.Format(Store, "%f11", "%r13", "SA") + "\n" + string.Format(Store, "%f13", "%r14", "SB") + "\n" + """
            add.u32 %r9, %r9, %nt;
            bra RP;
            RP_END:
            shl.b32 %r9, %r23, 1;
            add.u32 %r9, %r9, %tx;
            PT:
            setp.ge.u32 %p7, %r9, %s_cols;
            @%p7 bra PT_END;
            mul.wide.u32 %rd5, %r9, 4;
            add.u64 %rd7, %rd5, %rd2;
            ld.global.f32 %f2, [%rd7];
            mov.f32 %f5, 0f3F800000;
            add.u64 %rd7, %rd5, %b_gain;
            @%p11 ld.global.f32 %f5, [%rd7];
            @%p11 add.f32 %f5, %f5, %s_offset;
            mul.f32 %f2, %f2, %f1;
            mul.f32 %f2, %f2, %f5;
            """ + "\n" + string.Format(Store, "%f2", "%r9", "SC") + "\n" + """
            add.u32 %r9, %r9, %nt;
            bra PT;
            PT_END:
            """);

        // act(gate) and act'(gate) into %f5 and %f6 (kind 0 SiLU, 1 GELU tanh, 2 ReLU) from %f1.
        string activation = $"""
            setp.eq.u32 %p5, %s_kind, 1;
            @%p5 bra ACT_GELU;
            setp.eq.u32 %p5, %s_kind, 2;
            @%p5 bra ACT_RELU;
            mul.f32 %f3, %f1, {F(-1.4426950408889634f)};
            ex2.approx.ftz.f32 %f3, %f3;
            add.f32 %f3, %f3, {One};
            rcp.rn.f32 %f3, %f3;
            mul.f32 %f5, %f1, %f3;
            sub.f32 %f4, {One}, %f3;
            fma.rn.f32 %f4, %f1, %f4, {One};
            mul.f32 %f6, %f3, %f4;
            bra ACT_DONE;
            ACT_GELU:
            mul.f32 %f7, %f1, %f1;
            mul.f32 %f8, %f7, %f1;
            fma.rn.f32 %f8, %f8, {F(0.044715f)}, %f1;
            mul.f32 %f8, %f8, {F(0.7978845608f)};
            mul.f32 %f9, %f8, {F(2.8853900817779268f)};
            ex2.approx.ftz.f32 %f9, %f9;
            add.f32 %f9, %f9, {One};
            rcp.rn.f32 %f9, %f9;
            fma.rn.f32 %f9, %f9, {F(-2f)}, {One};
            add.f32 %f10, %f9, {One};
            mul.f32 %f5, %f10, %f1;
            mul.f32 %f5, %f5, {F(0.5f)};
            mul.f32 %f11, %f9, %f9;
            sub.f32 %f11, {One}, %f11;
            mul.f32 %f11, %f11, %f1;
            mul.f32 %f11, %f11, {F(0.5f * 0.7978845608f)};
            fma.rn.f32 %f12, %f7, {F(3f * 0.044715f)}, {One};
            mul.f32 %f11, %f11, %f12;
            fma.rn.f32 %f6, %f10, {F(0.5f)}, %f11;
            bra ACT_DONE;
            ACT_RELU:
            max.f32 %f5, %f1, {Zero};
            setp.gt.f32 %p6, %f1, {Zero};
            selp.f32 %f6, {One}, {Zero}, %p6;
            ACT_DONE:
            """;

        // y = act(gate) · up.
        Elementwise(sb, "gated_act_f32", ["gate", "up", "y"], [("u32", "kind")],
            """
            ld.global.f32 %f1, [%a_gate];
            ld.global.f32 %f2, [%a_up];
            """ + "\n" + activation + "\n" + """
            mul.f32 %f5, %f5, %f2;
            st.global.f32 [%a_y], %f5;
            """);

        // dgate += dy · up · act'(gate) (flags & 1), dup += dy · act(gate) (flags & 2); flags & 4 / 8: written (= instead of +=).
        Elementwise(sb, "gated_act_bwd_f32", ["gate", "up", "dy", "dgate", "dup"], [("u32", "kind"), ("u32", "flags")],
            """
            ld.global.f32 %f1, [%a_gate];
            ld.global.f32 %f2, [%a_up];
            ld.global.f32 %f13, [%a_dy];
            """ + "\n" + activation + "\n" + """
            and.b32 %r5, %s_flags, 1;
            setp.eq.u32 %p7, %r5, 0;
            @%p7 bra NO_DGATE;
            mul.f32 %f14, %f13, %f2;
            mov.f32 %f15, 0f00000000;
            and.b32 %r6, %s_flags, 4;
            setp.eq.u32 %p8, %r6, 0;
            @%p8 ld.global.f32 %f15, [%a_dgate];
            fma.rn.f32 %f15, %f14, %f6, %f15;
            st.global.f32 [%a_dgate], %f15;
            NO_DGATE:
            and.b32 %r5, %s_flags, 2;
            setp.eq.u32 %p7, %r5, 0;
            @%p7 bra NO_DUP;
            mov.f32 %f16, 0f00000000;
            and.b32 %r6, %s_flags, 8;
            setp.eq.u32 %p8, %r6, 0;
            @%p8 ld.global.f32 %f16, [%a_dup];
            fma.rn.f32 %f16, %f13, %f5, %f16;
            st.global.f32 [%a_dup], %f16;
            NO_DUP:
            """);

        // bfloat16 variants, one thread per word (elements 2i and 2i + 1, low half first, as bf16_pack_f32 writes them).
        // Round {f} to bfloat16 bits in the low half of {r} (to nearest, ties to even, in integer arithmetic).
        static string Round(string f, string r) => $"""
            mov.b32 {r}, {f};
            shr.u32 %r23, {r}, 16;
            and.b32 %r23, %r23, 1;
            add.u32 %r23, %r23, 32767;
            add.u32 {r}, {r}, %r23;
            shr.u32 {r}, {r}, 16;
            """;

        // Element e (%r11) of the pair: its index %r18, its float offset %rd1; past count: done.
        const string PairElement = """
            add.u32 %r18, %r10, %r11;
            setp.ge.u32 %p10, %r18, %s_count;
            @%p10 bra {0}_END;
            mul.wide.u32 %rd1, %r18, 4;
            shl.b32 %r19, %r11, 4;
            """;

        // gate, up (%f1, %f2) of element e from the packed words %r16, %r17.
        const string Unpack = """
            shr.u32 %r21, %r16, %r19;
            shl.b32 %r21, %r21, 16;
            mov.b32 %f1, %r21;
            shr.u32 %r21, %r17, %r19;
            shl.b32 %r21, %r21, 16;
            mov.b32 %f2, %r21;
            """;

        // y = act(gate) · up with gate and up read as floats or as packed bfloat16 (flags & 1: pg, pu), writing any of y
        // (flags & 4), gate and up packed (flags & 2) and y packed (flags & 8): the feed-forward activation that also
        // keeps its inputs and output as bfloat16 for the backward pass, or recomputes its output from them.
        Elementwise(sb, "gated_act_bf16_f32", ["gate", "up", "pg", "pu", "y", "py"], [("u32", "kind"), ("u32", "flags"), ("u32", "count")],
            """
            shl.b32 %r10, %i, 1;
            mov.u32 %r11, 0;
            mov.u32 %r12, 0;
            mov.u32 %r13, 0;
            mov.u32 %r14, 0;
            and.b32 %r15, %s_flags, 1;
            setp.ne.u32 %p9, %r15, 0;
            @%p9 ld.global.b32 %r16, [%a_pg];
            @%p9 ld.global.b32 %r17, [%a_pu];
            GA_LOOP:
            """ + "\n" + string.Format(PairElement, "GA") + "\n" + """
            @%p9 bra GA_PACKED;
            add.u64 %rd2, %b_gate, %rd1;
            ld.global.f32 %f1, [%rd2];
            add.u64 %rd2, %b_up, %rd1;
            ld.global.f32 %f2, [%rd2];
            bra GA_LOADED;
            GA_PACKED:
            """ + "\n" + Unpack + "\n" + """
            GA_LOADED:
            """ + "\n" + activation + "\n" + """
            mul.f32 %f5, %f5, %f2;
            and.b32 %r15, %s_flags, 4;
            setp.eq.u32 %p11, %r15, 0;
            @%p11 bra GA_NO_Y;
            add.u64 %rd2, %b_y, %rd1;
            st.global.f32 [%rd2], %f5;
            GA_NO_Y:
            and.b32 %r15, %s_flags, 2;
            setp.eq.u32 %p11, %r15, 0;
            @%p11 bra GA_NO_PACK_IN;
            """ + "\n" + Round("%f1", "%r22") + "\n" + """
            shl.b32 %r22, %r22, %r19;
            or.b32 %r12, %r12, %r22;
            """ + "\n" + Round("%f2", "%r22") + "\n" + """
            shl.b32 %r22, %r22, %r19;
            or.b32 %r13, %r13, %r22;
            GA_NO_PACK_IN:
            and.b32 %r15, %s_flags, 8;
            setp.eq.u32 %p11, %r15, 0;
            @%p11 bra GA_NO_PACK_Y;
            """ + "\n" + Round("%f5", "%r22") + "\n" + """
            shl.b32 %r22, %r22, %r19;
            or.b32 %r14, %r14, %r22;
            GA_NO_PACK_Y:
            add.u32 %r11, %r11, 1;
            setp.lt.u32 %p12, %r11, 2;
            @%p12 bra GA_LOOP;
            GA_END:
            and.b32 %r15, %s_flags, 2;
            setp.ne.u32 %p11, %r15, 0;
            @%p11 st.global.b32 [%a_pg], %r12;
            @%p11 st.global.b32 [%a_pu], %r13;
            and.b32 %r15, %s_flags, 8;
            setp.ne.u32 %p11, %r15, 0;
            @%p11 st.global.b32 [%a_py], %r14;
            """);

        // gated_act_bwd_f32 with gate and up read as packed bfloat16 words (pg, pu): no unpacked copy of either.
        Elementwise(sb, "gated_act_bwd_bf16_f32", ["pg", "pu", "dy", "dgate", "dup"], [("u32", "kind"), ("u32", "flags"), ("u32", "count")],
            """
            shl.b32 %r10, %i, 1;
            mov.u32 %r11, 0;
            ld.global.b32 %r16, [%a_pg];
            ld.global.b32 %r17, [%a_pu];
            GB_LOOP:
            """ + "\n" + string.Format(PairElement, "GB") + "\n" + Unpack + "\n" + """
            add.u64 %rd2, %b_dy, %rd1;
            ld.global.f32 %f13, [%rd2];
            """ + "\n" + activation + "\n" + """
            and.b32 %r15, %s_flags, 1;
            setp.eq.u32 %p11, %r15, 0;
            @%p11 bra GB_NO_DGATE;
            mul.f32 %f14, %f13, %f2;
            add.u64 %rd2, %b_dgate, %rd1;
            mov.f32 %f15, 0f00000000;
            and.b32 %r15, %s_flags, 4;
            setp.eq.u32 %p13, %r15, 0;
            @%p13 ld.global.f32 %f15, [%rd2];
            fma.rn.f32 %f15, %f14, %f6, %f15;
            st.global.f32 [%rd2], %f15;
            GB_NO_DGATE:
            and.b32 %r15, %s_flags, 2;
            setp.eq.u32 %p11, %r15, 0;
            @%p11 bra GB_NO_DUP;
            add.u64 %rd2, %b_dup, %rd1;
            mov.f32 %f16, 0f00000000;
            and.b32 %r15, %s_flags, 8;
            setp.eq.u32 %p13, %r15, 0;
            @%p13 ld.global.f32 %f16, [%rd2];
            fma.rn.f32 %f16, %f13, %f5, %f16;
            st.global.f32 [%rd2], %f16;
            GB_NO_DUP:
            add.u32 %r11, %r11, 1;
            setp.lt.u32 %p12, %r11, 2;
            @%p12 bra GB_LOOP;
            GB_END:
            """);

        // Rotary embedding of x [rows = batch·steps·heads, dim]: one thread per (row, pair). The pair of pair index p is
        // (2p, 2p+1) when interleaved, else (p, p + half). y must already hold x (dimensions beyond 2·half pass through).
        // sign = -1 applies the inverse rotation (the backward pass).
        Elementwise(sb, "rope_f32", ["x", "y", "cos", "sin", "positions"],
            [("u32", "heads"), ("u32", "steps"), ("u32", "dim"), ("u32", "half"), ("u32", "interleaved"), ("f32", "sign")], """
            div.u32 %r5, %i, %s_half;
            rem.u32 %r6, %i, %s_half;
            div.u32 %r7, %r5, %s_heads;
            rem.u32 %r7, %r7, %s_steps;
            mul.wide.u32 %rd1, %r7, 4;
            add.u64 %rd2, %b_positions, %rd1;
            ld.global.f32 %f1, [%rd2];
            cvt.rzi.u32.f32 %r8, %f1;
            mad.lo.u32 %r9, %r8, %s_half, %r6;
            mul.wide.u32 %rd3, %r9, 4;
            add.u64 %rd4, %b_cos, %rd3;
            ld.global.f32 %f2, [%rd4];
            add.u64 %rd4, %b_sin, %rd3;
            ld.global.f32 %f3, [%rd4];
            mul.f32 %f3, %f3, %s_sign;
            setp.ne.u32 %p1, %s_interleaved, 0;
            shl.b32 %r10, %r6, 1;
            add.u32 %r11, %r10, 1;
            add.u32 %r13, %r6, %s_half;
            selp.u32 %r14, %r10, %r6, %p1;
            selp.u32 %r15, %r11, %r13, %p1;
            mul.lo.u32 %r16, %r5, %s_dim;
            add.u32 %r14, %r14, %r16;
            add.u32 %r15, %r15, %r16;
            mul.wide.u32 %rd5, %r14, 4;
            mul.wide.u32 %rd6, %r15, 4;
            add.u64 %rd7, %b_x, %rd5;
            ld.global.f32 %f4, [%rd7];
            add.u64 %rd7, %b_x, %rd6;
            ld.global.f32 %f5, [%rd7];
            mul.f32 %f6, %f5, %f3;
            neg.f32 %f6, %f6;
            fma.rn.f32 %f6, %f4, %f2, %f6;
            mul.f32 %f7, %f4, %f3;
            fma.rn.f32 %f7, %f5, %f2, %f7;
            add.u64 %rd8, %b_y, %rd5;
            st.global.f32 [%rd8], %f6;
            add.u64 %rd8, %b_y, %rd6;
            st.global.f32 [%rd8], %f7;
            """);
    }
}
