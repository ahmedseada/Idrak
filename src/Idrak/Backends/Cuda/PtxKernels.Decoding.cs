// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;

namespace Idrak.Backends.Cuda;

// PTX for fused inference kernels and incremental decoding (KV cache, device-side positions, sampling).
internal static partial class PtxKernels
{
    public static readonly string[] DecodingNames =
    [
        "scale_mask_softmax_f32", "layernorm_fused_f32", "layernorm_train_f32", "layernorm_bwd_f32", "layernorm_bwd_params_f32", "bias_gelu_f32", "decoder_mask_f32", "kv_write_f32", "kv_write_bf16", "sample_rows_f32",
        "penalize_rows_f32", "history_push_f32", "sample_candidates_f32", "topk_candidates_f32",
    ];

    private const string PosInf = "0f7F800000";

    // LayerNorm for training: the forward pass of layernorm_fused_f32 that also stores mean and 1 / std per row; its
    // backward per row (dx += rstd · (g - mean(g) - x̂ · mean(g ∘ x̂)), g = dy ∘ gamma); and the gamma / beta gradients
    // (Σ over rows of dy ∘ x̂ and dy), one thread per column over a chunk of rows, added atomically.
    private static void LayerNormTraining(StringBuilder sb, string blockRowStart)
    {
        RowBlock(sb, "layernorm_train_f32", ["x", "gamma", "beta", "y", "stats"], [("u32", "cols"), ("f32", "eps")],
            blockRowStart + $"""
            add.u64 %rd2, %b_x, %rd1;
            add.u64 %rd3, %b_y, %rd1;
            sub.u64 %rd6, %rd3, %rd2;
            cvt.rn.f32.u32 %f10, %s_cols;
            mov.f32 %f1, {Zero};
            """ + "\n" + StridedLoop("LS", "%rd2", "%s_cols", "add.f32 %f1, %f1, %f2;") + "\n"
            + BlockReduce("RMEAN", "%f1", "add", Zero) + "\n" + $"""
            div.rn.f32 %f3, %f1, %f10;
            mov.f32 %f4, {Zero};
            """ + "\n" + StridedLoop("LV", "%rd2", "%s_cols", """
            sub.f32 %f5, %f2, %f3;
            fma.rn.f32 %f4, %f5, %f5, %f4;
            """) + "\n" + BlockReduce("RVAR", "%f4", "add", Zero) + "\n" + """
            div.rn.f32 %f4, %f4, %f10;
            add.f32 %f4, %f4, %s_eps;
            sqrt.rn.f32 %f4, %f4;
            rcp.rn.f32 %f4, %f4;
            setp.eq.u32 %p5, %tx, 0;
            @%p5 st.global.f32 [%a_stats], %f3;
            mul.wide.u32 %rd11, %n, 4;
            add.u64 %rd11, %rd11, %a_stats;
            @%p5 st.global.f32 [%rd11], %f4;
            """ + "\n" + StridedLoop("LW", "%rd2", "%s_cols", """
            mul.wide.u32 %rd7, %r6, 4;
            add.u64 %rd8, %b_gamma, %rd7;
            ld.global.f32 %f6, [%rd8];
            add.u64 %rd9, %b_beta, %rd7;
            ld.global.f32 %f7, [%rd9];
            sub.f32 %f5, %f2, %f3;
            mul.f32 %f5, %f5, %f4;
            fma.rn.f32 %f5, %f5, %f6, %f7;
            add.u64 %rd10, %rd5, %rd6;
            st.global.f32 [%rd10], %f5;
            """));

        // dx: %f3 = mean, %f4 = rstd; pass 1 sums g and g·x̂, pass 2 adds rstd · (g - s1 / cols - x̂ · s2 / cols).
        RowBlock(sb, "layernorm_bwd_f32", ["x", "gamma", "dy", "stats", "dx"], [("u32", "cols")],
            blockRowStart + $"""
            add.u64 %rd2, %b_x, %rd1;
            add.u64 %rd3, %b_dy, %rd1;
            add.u64 %rd4, %b_dx, %rd1;
            sub.u64 %rd6, %rd3, %rd2;
            sub.u64 %rd12, %rd4, %rd2;
            ld.global.f32 %f3, [%a_stats];
            mul.wide.u32 %rd11, %n, 4;
            add.u64 %rd11, %rd11, %a_stats;
            ld.global.f32 %f4, [%rd11];
            cvt.rn.f32.u32 %f10, %s_cols;
            rcp.rn.f32 %f10, %f10;
            mov.f32 %f11, {Zero};
            mov.f32 %f12, {Zero};
            """ + "\n" + StridedLoop("B1", "%rd2", "%s_cols", """
            mul.wide.u32 %rd7, %r6, 4;
            add.u64 %rd8, %b_gamma, %rd7;
            ld.global.f32 %f6, [%rd8];
            add.u64 %rd9, %rd5, %rd6;
            ld.global.f32 %f7, [%rd9];
            mul.f32 %f7, %f7, %f6;
            sub.f32 %f5, %f2, %f3;
            mul.f32 %f5, %f5, %f4;
            add.f32 %f11, %f11, %f7;
            fma.rn.f32 %f12, %f7, %f5, %f12;
            """) + "\n" + BlockReduce("R1", "%f11", "add", Zero) + "\n" + BlockReduce("R2", "%f12", "add", Zero) + "\n" + """
            mul.f32 %f11, %f11, %f10;
            mul.f32 %f12, %f12, %f10;
            """ + "\n" + StridedLoop("B2", "%rd2", "%s_cols", """
            mul.wide.u32 %rd7, %r6, 4;
            add.u64 %rd8, %b_gamma, %rd7;
            ld.global.f32 %f6, [%rd8];
            add.u64 %rd9, %rd5, %rd6;
            ld.global.f32 %f7, [%rd9];
            mul.f32 %f7, %f7, %f6;
            sub.f32 %f5, %f2, %f3;
            mul.f32 %f5, %f5, %f4;
            sub.f32 %f7, %f7, %f11;
            mul.f32 %f8, %f5, %f12;
            sub.f32 %f7, %f7, %f8;
            add.u64 %rd9, %rd5, %rd12;
            ld.global.f32 %f9, [%rd9];
            fma.rn.f32 %f9, %f7, %f4, %f9;
            st.global.f32 [%rd9], %f9;
            """));

        // dgamma / dbeta: thread = column (x = column block of BlockSize), grid y = chunks of `chunk` rows.
        sb.AppendLine($$"""
            .visible .entry layernorm_bwd_params_f32(
                .param .u64 p_x, .param .u64 p_dy, .param .u64 p_stats, .param .u64 p_dgamma, .param .u64 p_dbeta,
                .param .u32 p_rows, .param .u32 p_cols, .param .u32 p_chunk
            )
            {
                .reg .pred %p<4>;
                .reg .b32 %r<16>;
                .reg .b64 %rd<16>;
                .reg .f32 %f<12>;
                ld.param.u64 %rd1, [p_x];
                ld.param.u64 %rd2, [p_dy];
                ld.param.u64 %rd3, [p_stats];
                ld.param.u64 %rd4, [p_dgamma];
                ld.param.u64 %rd5, [p_dbeta];
                cvta.to.global.u64 %rd1, %rd1;
                cvta.to.global.u64 %rd2, %rd2;
                cvta.to.global.u64 %rd3, %rd3;
                ld.param.u32 %r1, [p_rows];
                ld.param.u32 %r2, [p_cols];
                ld.param.u32 %r3, [p_chunk];
                mov.u32 %r4, %ctaid.x;
                mov.u32 %r5, %tid.x;
                mad.lo.u32 %r6, %r4, {{BlockSize}}, %r5;
                setp.ge.u32 %p1, %r6, %r2;
                @%p1 bra DONE;
                mov.u32 %r7, %ctaid.y;
                mul.lo.u32 %r8, %r7, %r3;
                add.u32 %r9, %r8, %r3;
                min.u32 %r9, %r9, %r1;
                mov.f32 %f1, 0f00000000;
                mov.f32 %f2, 0f00000000;
                mul.wide.u32 %rd12, %r1, 4;
            ROWS:
                setp.ge.u32 %p2, %r8, %r9;
                @%p2 bra ROWS_END;
                mad.lo.u32 %r10, %r8, %r2, %r6;
                mul.wide.u32 %rd6, %r10, 4;
                add.u64 %rd7, %rd6, %rd1;
                add.u64 %rd8, %rd6, %rd2;
                ld.global.f32 %f3, [%rd7];
                ld.global.f32 %f4, [%rd8];
                mul.wide.u32 %rd9, %r8, 4;
                add.u64 %rd10, %rd9, %rd3;
                ld.global.f32 %f5, [%rd10];
                add.u64 %rd11, %rd10, %rd12;
                ld.global.f32 %f6, [%rd11];
                sub.f32 %f3, %f3, %f5;
                mul.f32 %f3, %f3, %f6;
                fma.rn.f32 %f1, %f4, %f3, %f1;
                add.f32 %f2, %f2, %f4;
                add.u32 %r8, %r8, 1;
                bra ROWS;
            ROWS_END:
                mul.wide.u32 %rd13, %r6, 4;
                setp.eq.u64 %p3, %rd4, 0;
                @%p3 bra NO_GAMMA;
                cvta.to.global.u64 %rd4, %rd4;
                add.u64 %rd14, %rd4, %rd13;
                red.global.add.f32 [%rd14], %f1;
            NO_GAMMA:
                setp.eq.u64 %p3, %rd5, 0;
                @%p3 bra DONE;
                cvta.to.global.u64 %rd5, %rd5;
                add.u64 %rd15, %rd5, %rd13;
                red.global.add.f32 [%rd15], %f2;
            DONE:
                ret;
            }
            """);
    }

    private static void BuildDecoding(StringBuilder sb)
    {
        const string BlockRowStart = """
            mul.lo.u32 %r5, %row, %s_cols;
            mul.wide.u32 %rd1, %r5, 4;
            """;

        // softmax(scale * x + mask[row % maskrows]) per row, one block per row.
        RowBlock(sb, "scale_mask_softmax_f32", ["x", "mask", "y"], [("u32", "cols"), ("u32", "maskrows"), ("f32", "scale"), ("u32", "hasmask")],
            BlockRowStart + $"""
            add.u64 %rd2, %b_x, %rd1;
            add.u64 %rd3, %b_y, %rd1;
            sub.u64 %rd6, %rd3, %rd2;
            rem.u32 %r7, %row, %s_maskrows;
            mul.lo.u32 %r7, %r7, %s_cols;
            mul.wide.u32 %rd4, %r7, 4;
            add.u64 %rd4, %b_mask, %rd4;
            sub.u64 %rd8, %rd4, %rd2;
            setp.ne.u32 %p3, %s_hasmask, 0;
            mov.f32 %f1, {NegInf};
            """ + "\n" + StridedLoop("SZ", "%rd2", "%s_cols", """
            mul.f32 %f2, %f2, %s_scale;
            mov.f32 %f3, 0f00000000;
            add.u64 %rd9, %rd5, %rd8;
            @%p3 ld.global.f32 %f3, [%rd9];
            add.f32 %f2, %f2, %f3;
            add.u64 %rd7, %rd5, %rd6;
            st.global.f32 [%rd7], %f2;
            max.f32 %f1, %f1, %f2;
            """) + "\n" + BlockReduce("RMAX", "%f1", "max", NegInf) + "\n" + $"""
            mov.f32 %f4, {Zero};
            """ + "\n" + StridedLoop("SE", "%rd3", "%s_cols", $"""
            sub.f32 %f2, %f2, %f1;
            mul.f32 %f2, %f2, {Log2E};
            ex2.approx.ftz.f32 %f2, %f2;
            st.global.f32 [%rd5], %f2;
            add.f32 %f4, %f4, %f2;
            """) + "\n" + BlockReduce("RSUM", "%f4", "add", Zero) + "\n" + """
            rcp.rn.f32 %f4, %f4;
            """ + "\n" + StridedLoop("SN", "%rd3", "%s_cols", """
            mul.f32 %f2, %f2, %f4;
            st.global.f32 [%rd5], %f2;
            """));

        LayerNormTraining(sb, BlockRowStart);

        // LayerNorm over the last dimension with gamma/beta, one block per row.
        RowBlock(sb, "layernorm_fused_f32", ["x", "gamma", "beta", "y"], [("u32", "cols"), ("f32", "eps")],
            BlockRowStart + $"""
            add.u64 %rd2, %b_x, %rd1;
            add.u64 %rd3, %b_y, %rd1;
            sub.u64 %rd6, %rd3, %rd2;
            cvt.rn.f32.u32 %f10, %s_cols;
            mov.f32 %f1, {Zero};
            """ + "\n" + StridedLoop("LS", "%rd2", "%s_cols", "add.f32 %f1, %f1, %f2;") + "\n"
            + BlockReduce("RMEAN", "%f1", "add", Zero) + "\n" + $"""
            div.rn.f32 %f3, %f1, %f10;
            mov.f32 %f4, {Zero};
            """ + "\n" + StridedLoop("LV", "%rd2", "%s_cols", """
            sub.f32 %f5, %f2, %f3;
            fma.rn.f32 %f4, %f5, %f5, %f4;
            """) + "\n" + BlockReduce("RVAR", "%f4", "add", Zero) + "\n" + """
            div.rn.f32 %f4, %f4, %f10;
            add.f32 %f4, %f4, %s_eps;
            sqrt.rn.f32 %f4, %f4;
            rcp.rn.f32 %f4, %f4;
            """ + "\n" + StridedLoop("LW", "%rd2", "%s_cols", """
            mul.wide.u32 %rd7, %r6, 4;
            add.u64 %rd8, %b_gamma, %rd7;
            ld.global.f32 %f6, [%rd8];
            add.u64 %rd9, %b_beta, %rd7;
            ld.global.f32 %f7, [%rd9];
            sub.f32 %f5, %f2, %f3;
            mul.f32 %f5, %f5, %f4;
            fma.rn.f32 %f5, %f5, %f6, %f7;
            add.u64 %rd10, %rd5, %rd6;
            st.global.f32 [%rd10], %f5;
            """));

        // gelu(x + bias[i % cols]).
        Elementwise(sb, "bias_gelu_f32", ["x", "bias", "y"], [("u32", "cols")],
            $"""
            rem.u32 %r5, %i, %s_cols;
            mul.wide.u32 %rd1, %r5, 4;
            add.u64 %rd2, %b_bias, %rd1;
            ld.global.f32 %f1, [%a_x];
            ld.global.f32 %f2, [%rd2];
            add.f32 %f1, %f1, %f2;
            mul.f32 %f2, %f1, %f1;
            mul.f32 %f3, %f2, %f1;
            fma.rn.f32 %f4, %f3, {F(0.044715f)}, %f1;
            mul.f32 %f4, %f4, {F(0.7978845608f)};
            mul.f32 %f5, %f4, {F(2.8853900817779268f)};
            ex2.approx.ftz.f32 %f5, %f5;
            add.f32 %f5, %f5, {One};
            rcp.rn.f32 %f5, %f5;
            fma.rn.f32 %f6, %f5, {F(-2f)}, {One};
            add.f32 %f7, %f6, {One};
            mul.f32 %f7, %f7, %f1;
            mul.f32 %f7, %f7, {F(0.5f)};
            st.global.f32 [%a_y], %f7;
            """);

        // mask[i, j] = j <= pos + i ? 0 : -1e9 with pos read from device memory.
        Elementwise(sb, "decoder_mask_f32", ["pos", "mask"], [("u32", "cap")],
            $"""
            ld.global.f32 %f1, [%b_pos];
            cvt.rzi.u32.f32 %r5, %f1;
            div.u32 %r6, %i, %s_cap;
            rem.u32 %r7, %i, %s_cap;
            add.u32 %r8, %r5, %r6;
            setp.le.u32 %p1, %r7, %r8;
            selp.f32 %f2, {Zero}, {F(-1e9f)}, %p1;
            st.global.f32 [%a_mask], %f2;
            """);

        // bfloat16 cache: word w of row (h, pos + t) = the rounded pair src[h, t, 2w], src[h, t, 2w + 1] (0 past dim),
        // rounded to nearest, ties to even; one thread per word.
        Elementwise(sb, "kv_write_bf16", ["src", "cache", "pos"], [("u32", "steps"), ("u32", "cap"), ("u32", "dim"), ("u32", "words")],
            """
            ld.global.f32 %f1, [%b_pos];
            cvt.rzi.u32.f32 %r5, %f1;
            div.u32 %r6, %i, %s_words;
            rem.u32 %r7, %i, %s_words;
            div.u32 %r8, %r6, %s_steps;
            rem.u32 %r9, %r6, %s_steps;
            mad.lo.u32 %r10, %r8, %s_cap, %r5;
            add.u32 %r10, %r10, %r9;
            mad.lo.u32 %r10, %r10, %s_words, %r7;
            mul.wide.u32 %rd1, %r10, 4;
            add.u64 %rd1, %b_cache, %rd1;
            shl.b32 %r11, %r7, 1;
            mad.lo.u32 %r12, %r6, %s_dim, %r11;
            mul.wide.u32 %rd2, %r12, 4;
            add.u64 %rd2, %b_src, %rd2;
            ld.global.u32 %r13, [%rd2];
            shr.u32 %r14, %r13, 16;
            and.b32 %r14, %r14, 1;
            add.u32 %r14, %r14, 0x7FFF;
            add.u32 %r13, %r13, %r14;
            shr.u32 %r13, %r13, 16;
            add.u32 %r15, %r11, 1;
            mov.u32 %r16, 0;
            setp.lt.u32 %p1, %r15, %s_dim;
            @%p1 ld.global.u32 %r16, [%rd2+4];
            shr.u32 %r14, %r16, 16;
            and.b32 %r14, %r14, 1;
            add.u32 %r14, %r14, 0x7FFF;
            add.u32 %r16, %r16, %r14;
            and.b32 %r16, %r16, 0xFFFF0000;
            or.b32 %r13, %r13, %r16;
            st.global.u32 [%rd1], %r13;
            """);

        // cache[h, pos + t, d] = src[h, t, d]; one thread per source element.
        Elementwise(sb, "kv_write_f32", ["src", "cache", "pos"], [("u32", "stepsdim"), ("u32", "capdim"), ("u32", "dim")],
            """
            ld.global.f32 %f1, [%b_pos];
            cvt.rzi.u32.f32 %r5, %f1;
            div.u32 %r6, %i, %s_stepsdim;
            rem.u32 %r7, %i, %s_stepsdim;
            mul.lo.u32 %r8, %r6, %s_capdim;
            mad.lo.u32 %r8, %r5, %s_dim, %r8;
            add.u32 %r8, %r8, %r7;
            mul.wide.u32 %rd1, %r8, 4;
            add.u64 %rd1, %b_cache, %rd1;
            ld.global.f32 %f2, [%a_src];
            st.global.f32 [%rd1], %f2;
            """);

        SampleRows(sb);
    }

    /// <summary>
    /// The sampler: see Backend.SampleRows; one block per row, mirroring CpuBackend.SampleRows step by step (each pass
    /// over the vocabulary is a block reduction). The draw: each thread sums the kept weights of a contiguous chunk,
    /// thread 0 finds the chunk where the running total passes the target, and that chunk's thread walks it.
    /// </summary>
    private static string SamplerBody(bool candidates)
    {
        // Strided loop over the row's vocabulary: %r6 = index, %rd3 = address, %f2 = scaled score, %f5 = kept weight.
        static string Kept(string label, string body) => $"""
            mov.u32 %r6, %tx;
            {label}:
            setp.ge.u32 %p1, %r6, %r29;
            @%p1 bra {label}_END;
            mul.wide.u32 %rd3, %r6, 4;
            add.u64 %rd3, %rd3, %rd2;
            ld.global.f32 %f2, [%rd3];
            mul.f32 %f2, %f2, %f31;
            setp.ge.f32 %p4, %f2, %f3;
            sub.f32 %f5, %f2, %f1;
            mul.f32 %f5, %f5, {Log2E};
            ex2.approx.ftz.f32 %f5, %f5;
            selp.f32 %f5, %f5, {Zero}, %p4;
            {body}
            {label}_NEXT:
            add.u32 %r6, %r6, %nt;
            bra {label};
            {label}_END:
            """;

        // The same over this thread's chunk [%r21, %r22).
        static string Chunk(string label, string body) => $"""
            mov.u32 %r6, %r21;
            {label}:
            setp.ge.u32 %p1, %r6, %r22;
            @%p1 bra {label}_END;
            mul.wide.u32 %rd3, %r6, 4;
            add.u64 %rd3, %rd3, %rd2;
            ld.global.f32 %f2, [%rd3];
            mul.f32 %f2, %f2, %f31;
            setp.ge.f32 %p4, %f2, %f3;
            sub.f32 %f5, %f2, %f1;
            mul.f32 %f5, %f5, {Log2E};
            ex2.approx.ftz.f32 %f5, %f5;
            selp.f32 %f5, %f5, {Zero}, %p4;
            {body}
            {label}_NEXT:
            add.u32 %r6, %r6, 1;
            bra {label};
            {label}_END:
            """;

        var body = new StringBuilder();
        // %rd2 = the scores walked (the logits row, or with candidates the row's candidate scores, already divided by
        // the temperature), %r29 = how many, %f31 = the factor applied to each, %p14 = ids map through candidate indices.
        if (candidates)
        {
            body.AppendLine($"""
                mul.wide.u32 %rd1, %row, 4;
                add.u64 %rd6, %b_flags, %rd1;
                ld.global.u32 %r28, [%rd6];
                setp.ne.u32 %p14, %r28, 0xFFFFFFFF;
                @!%p14 bra FULL_ROW;
                mul.lo.u32 %r5, %row, %s_slots;
                mul.wide.u32 %rd1, %r5, 4;
                add.u64 %rd2, %b_candv, %rd1;
                add.u64 %rd8, %b_candi, %rd1;
                mov.u32 %r29, %s_slots;
                mov.f32 %f31, {One};
                bra SOURCE_DONE;
                FULL_ROW:
                setp.ne.u32 %p14, 0, 0;
                """);
        }
        else
        {
            body.AppendLine("setp.ne.u32 %p14, 0, 0;");
        }

        body.AppendLine($"""
            mad.lo.u32 %r5, %row, %s_rowstride, %s_rowoffset;
            mul.wide.u32 %rd1, %r5, 4;
            add.u64 %rd2, %b_logits, %rd1;
            mov.u32 %r29, %s_vocab;
            mov.f32 %f31, %s_invt;
            SOURCE_DONE:
            mov.f32 %f1, {NegInf};
            mov.f32 %f3, {NegInf};
            """);
        body.AppendLine(Kept("PMAX", "max.f32 %f1, %f1, %f2;"));
        body.AppendLine(BlockReduce("RMAX", "%f1", "max", NegInf));
        body.AppendLine($"""
            mov.f32 %f3, {NegInf};
            setp.eq.u32 %p2, %s_topk, 0;
            @%p2 bra THR_DONE;
            setp.ge.u32 %p2, %s_topk, %r29;
            @%p2 bra THR_DONE;
            mov.f32 %f3, {PosInf};
            mov.u32 %r7, 0;
            TK:
            setp.ge.u32 %p2, %r7, %s_topk;
            @%p2 bra THR_DONE;
            mov.f32 %f4, {NegInf};
            """);
        body.AppendLine(Kept("TKI", """
            setp.lt.f32 %p3, %f2, %f3;
            setp.gt.and.f32 %p3, %f2, %f4, %p3;
            selp.f32 %f4, %f2, %f4, %p3;
            """));
        body.AppendLine(BlockReduce("RTK", "%f4", "max", NegInf));
        body.AppendLine($"""
            mov.f32 %f3, %f4;
            add.u32 %r7, %r7, 1;
            bra TK;
            THR_DONE:
            """);
        if (candidates)
        {
            // A slice that dropped tied scores recorded the largest one dropped (%r28, order-preserving bits): the
            // candidates are complete only when the top-k threshold is above it; otherwise start over on the full row.
            body.AppendLine("""
                @!%p14 bra COMPLETE;
                setp.eq.u32 %p2, %r28, 0;
                @%p2 bra COMPLETE;
                mov.b32 %r24, %f3;
                shr.s32 %r25, %r24, 31;
                or.b32 %r25, %r25, 0x80000000;
                xor.b32 %r24, %r24, %r25;
                setp.gt.u32 %p2, %r24, %r28;
                @!%p2 bra FULL_ROW;
                COMPLETE:
                """);
        }

        body.AppendLine($"""
            setp.gt.f32 %p7, %s_minp, {Zero};
            @!%p7 bra MINP_DONE;
            lg2.approx.ftz.f32 %f19, %s_minp;
            mul.f32 %f19, %f19, {F(0.6931471805599453f)};
            add.f32 %f19, %f19, %f1;
            max.f32 %f3, %f3, %f19;
            MINP_DONE:
            setp.lt.f32 %p7, %s_topp, {One};
            setp.gt.and.f32 %p7, %s_topp, {Zero}, %p7;
            @!%p7 bra TOPP_DONE;
            mov.f32 %f20, {Zero};
            """);
        body.AppendLine(Kept("TPM", "add.f32 %f20, %f20, %f5;"));
        body.AppendLine(BlockReduce("RTPM", "%f20", "add", Zero));
        body.AppendLine($"""
            mul.f32 %f20, %f20, %s_topp;
            sub.f32 %f21, %f1, {F(40f)};
            max.f32 %f21, %f21, %f3;
            mov.f32 %f22, %f1;
            mov.u32 %r19, 0;
            BIS:
            setp.ge.u32 %p8, %r19, 24;
            @%p8 bra BIS_END;
            add.f32 %f23, %f21, %f22;
            mul.f32 %f23, %f23, {F(0.5f)};
            mov.f32 %f24, {Zero};
            """);
        body.AppendLine(Kept("TPB", $"""
            setp.ge.f32 %p9, %f2, %f23;
            selp.f32 %f25, %f5, {Zero}, %p9;
            add.f32 %f24, %f24, %f25;
            """));
        body.AppendLine(BlockReduce("RTPB", "%f24", "add", Zero));
        body.AppendLine($"""
            setp.ge.f32 %p9, %f24, %f20;
            selp.f32 %f21, %f23, %f21, %p9;
            selp.f32 %f22, %f22, %f23, %p9;
            add.u32 %r19, %r19, 1;
            bra BIS;
            BIS_END:
            mov.f32 %f3, %f21;
            TOPP_DONE:
            mov.f32 %f6, {Zero};
            """);
        body.AppendLine(Kept("PSUM", "add.f32 %f6, %f6, %f5;"));
        body.AppendLine(BlockReduce("RPSUM", "%f6", "add", Zero));

        // The target: uniform(seed, step, row) · Σ weights (the counter-based random stream of the CPU sampler).
        body.AppendLine($"""
            ld.global.f32 %f7, [%b_step];
            cvt.rzi.u32.f32 %r8, %f7;
            mul.lo.u32 %r9, %r8, 0x9E3779B9;
            xor.b32 %r9, %r9, %s_seed;
            mul.lo.u32 %r10, %i, 0x85EBCA6B;
            xor.b32 %r9, %r9, %r10;
            shr.u32 %r10, %r9, 16;
            xor.b32 %r9, %r9, %r10;
            mul.lo.u32 %r9, %r9, 0x85EBCA6B;
            shr.u32 %r10, %r9, 13;
            xor.b32 %r9, %r9, %r10;
            mul.lo.u32 %r9, %r9, 0xC2B2AE35;
            shr.u32 %r10, %r9, 16;
            xor.b32 %r9, %r9, %r10;
            shr.u32 %r10, %r9, 8;
            cvt.rn.f32.u32 %f8, %r10;
            mul.f32 %f8, %f8, {F(1f / 16777216f)};
            mul.f32 %f8, %f8, %f6;
            add.u32 %r20, %r29, %nt;
            sub.u32 %r20, %r20, 1;
            div.u32 %r20, %r20, %nt;
            mul.lo.u32 %r21, %tx, %r20;
            min.u32 %r21, %r21, %r29;
            add.u32 %r22, %r21, %r20;
            min.u32 %r22, %r22, %r29;
            mov.f32 %f26, {Zero};
            """);
        body.AppendLine(Chunk("PC", "add.f32 %f26, %f26, %f5;"));
        body.AppendLine($"""
            shl.b32 %r23, %tx, 2;
            add.u32 %r23, %r23, %spart;
            st.shared.f32 [%r23], %f26;
            bar.sync 0;
            setp.ne.u32 %p10, %tx, 0;
            @%p10 bra SCAN_DONE;
            mov.f32 %f27, {Zero};
            mov.s32 %r24, -1;
            mov.f32 %f28, {Zero};
            mov.u32 %r25, 0;
            SC:
            setp.ge.u32 %p11, %r25, %nt;
            @%p11 bra SC_END;
            shl.b32 %r23, %r25, 2;
            add.u32 %r23, %r23, %spart;
            ld.shared.f32 %f29, [%r23];
            setp.gt.f32 %p12, %f29, {Zero};
            @!%p12 bra SC_NEXT;
            mov.b32 %r24, %r25;
            mov.f32 %f28, %f27;
            add.f32 %f27, %f27, %f29;
            setp.gt.f32 %p12, %f27, %f8;
            @%p12 bra SC_END;
            SC_NEXT:
            add.u32 %r25, %r25, 1;
            bra SC;
            SC_END:
            st.shared.b32 [%sb], %r24;
            st.shared.f32 [%sb+4], %f28;
            SCAN_DONE:
            bar.sync 0;
            ld.shared.b32 %r24, [%sb];
            ld.shared.f32 %f28, [%sb+4];
            bar.sync 0;
            mov.s32 %r11, -1;
            mov.f32 %f10, {Zero};
            setp.ne.u32 %p10, %tx, %r24;
            @%p10 bra PICK_DONE;
            mov.f32 %f9, %f28;
            """);
        body.AppendLine(Chunk("PS", """
            setp.gt.f32 %p5, %f5, 0f00000000;
            @!%p5 bra PS_NEXT;
            mov.u32 %r11, %r6;
            mov.f32 %f10, %f5;
            add.f32 %f9, %f9, %f5;
            setp.gt.f32 %p6, %f9, %f8;
            @%p6 bra PS_END;
            """));
        body.AppendLine("""
            st.shared.b32 [%sb], %r11;
            st.shared.f32 [%sb+4], %f10;
            PICK_DONE:
            bar.sync 0;
            ld.shared.b32 %r11, [%sb];
            ld.shared.f32 %f10, [%sb+4];
            bar.sync 0;
            mov.f32 %f11, 0f00000000;
            """);
        body.AppendLine(Kept("PH", """
            setp.gt.f32 %p5, %f5, 0f00000000;
            @!%p5 bra PH_NEXT;
            div.rn.f32 %f12, %f5, %f6;
            lg2.approx.ftz.f32 %f13, %f12;
            mul.f32 %f13, %f13, %f12;
            sub.f32 %f11, %f11, %f13;
            """));
        body.AppendLine(BlockReduce("RPH", "%f11", "add", Zero));
        body.AppendLine("""
            setp.eq.u32 %p15, %tx, 0;
            mad.lo.u32 %r12, %r8, %s_rows, %i;
            mul.lo.u32 %r12, %r12, 13;
            mul.wide.u32 %rd4, %r12, 4;
            add.u64 %rd5, %b_stats, %rd4;
            setp.ge.s32 %p13, %r11, 0;
            and.pred %p13, %p13, %p14;
            mul.wide.s32 %rd9, %r11, 4;
            add.u64 %rd9, %rd9, %rd8;
            mov.b32 %r26, %r11;
            @%p13 ld.global.f32 %f30, [%rd9];
            @%p13 cvt.rzi.s32.f32 %r26, %f30;
            cvt.rn.f32.s32 %f14, %r26;
            @%p15 st.global.f32 [%a_ids], %f14;
            @%p15 st.global.f32 [%rd5], %f14;
            div.rn.f32 %f15, %f10, %f6;
            @%p15 st.global.f32 [%rd5+4], %f15;
            @%p15 st.global.f32 [%rd5+8], %f11;
            mov.s32 %r13, -1;
            mov.s32 %r14, -1;
            mov.s32 %r15, -1;
            mov.s32 %r16, -1;
            mov.s32 %r17, -1;
            """);
        for (int a = 0; a < 5; a++)
        {
            var exclude = new StringBuilder("setp.gt.f32 %p2, %f5, %f16;\n");
            for (int prev = 0; prev < a; prev++)
            {
                exclude.AppendLine($"setp.ne.s32 %p3, %r6, %r{13 + prev};");
                exclude.AppendLine("and.pred %p2, %p2, %p3;");
            }

            exclude.AppendLine("selp.f32 %f16, %f5, %f16, %p2;");
            exclude.Append("selp.b32 %r18, %r6, %r18, %p2;");
            body.AppendLine($"""
                mov.s32 %r18, -1;
                mov.f32 %f16, {Zero};
                """);
            body.AppendLine(Kept($"TOP{a}", exclude.ToString()));
            body.AppendLine(BlockArgMax($"RTOP{a}", "%f16", "%r18"));
            body.AppendLine($"""
                mov.b32 %r{13 + a}, %r18;
                setp.ge.s32 %p13, %r18, 0;
                and.pred %p13, %p13, %p14;
                mul.wide.s32 %rd9, %r18, 4;
                add.u64 %rd9, %rd9, %rd8;
                mov.b32 %r26, %r18;
                @%p13 ld.global.f32 %f30, [%rd9];
                @%p13 cvt.rzi.s32.f32 %r26, %f30;
                cvt.rn.f32.s32 %f17, %r26;
                @%p15 st.global.f32 [%rd5+{12 + 8 * a}], %f17;
                div.rn.f32 %f18, %f16, %f6;
                @%p15 st.global.f32 [%rd5+{16 + 8 * a}], %f18;
                """);
        }

        return body.ToString();
    }

    private static void SampleRows(StringBuilder sb)
    {
        RowBlock(sb, "sample_rows_f32", ["logits", "ids", "stats", "step"],
            [("u32", "vocab"), ("u32", "rowstride"), ("u32", "rowoffset"), ("f32", "invt"), ("u32", "topk"), ("f32", "topp"), ("f32", "minp"), ("u32", "seed"), ("u32", "rows")],
            SamplerBody(candidates: false), sharedFloats: SamplerThreads);
        RowBlock(sb, "sample_candidates_f32", ["logits", "ids", "stats", "step", "candv", "candi", "flags"],
            [("u32", "vocab"), ("u32", "rowstride"), ("u32", "rowoffset"), ("f32", "invt"), ("u32", "topk"), ("f32", "topp"), ("f32", "minp"), ("u32", "seed"), ("u32", "rows"), ("u32", "slots")],
            SamplerBody(candidates: true), sharedFloats: SamplerThreads);
        TopKCandidates(sb);
        PenaltyKernels(sb);
    }

    /// <summary>Scores per <c>topk_candidates_f32</c> block (8 per thread).</summary>
    public const int CandidateSlice = 2048;

    /// <summary>Candidate slots per <c>topk_candidates_f32</c> block.</summary>
    public const int CandidateSlots = 64;

    // Stage one of top-k sampling (topk ≤ 64): block b of row r takes scores [2048·b, 2048·b + 2048), finds its k-th
    // largest distinct scaled score and writes the scores at or above it (with their token ids, in vocabulary order)
    // to candv/candi[(r·blocks + b)·64 …], filling unused slots with -inf. The global k-th largest distinct score is
    // at or above every block's, so the candidates hold every token top-k keeps. A block with more than 64 (ties)
    // sets flags[r], and sample_candidates_f32 then walks the whole row. Grid: rows · blocks, one block each.
    private static void TopKCandidates(StringBuilder sb)
    {
        var b = new StringBuilder();
        b.AppendLine($"""
            div.u32 %r7, %row, %s_bpr;
            rem.u32 %r8, %row, %s_bpr;
            mad.lo.u32 %r5, %r7, %s_rowstride, %s_rowoffset;
            mul.wide.u32 %rd1, %r5, 4;
            add.u64 %rd2, %b_logits, %rd1;
            shl.b32 %r9, %r8, {(int)Math.Log2(CandidateSlice)};
            shl.b32 %r10, %tx, 3;
            add.u32 %r10, %r10, %r9;
            """);
        for (int i = 0; i < 8; i++)
        {
            b.AppendLine($"""
                add.u32 %r11, %r10, {i};
                setp.lt.u32 %p{1 + i}, %r11, %s_vocab;
                mov.f32 %f{10 + i}, {NegInf};
                mul.wide.u32 %rd3, %r11, 4;
                add.u64 %rd3, %rd3, %rd2;
                @%p{1 + i} ld.global.f32 %f{10 + i}, [%rd3];
                @%p{1 + i} mul.f32 %f{10 + i}, %f{10 + i}, %s_invt;
                """);
        }

        b.AppendLine($"""
            mov.f32 %f3, {PosInf};
            mov.u32 %r12, 0;
            TK:
            setp.ge.u32 %p9, %r12, %s_topk;
            @%p9 bra TK_END;
            mov.f32 %f4, {NegInf};
            """);
        for (int i = 0; i < 8; i++)
        {
            b.AppendLine($"""
                setp.lt.f32 %p10, %f{10 + i}, %f3;
                setp.gt.and.f32 %p10, %f{10 + i}, %f4, %p10;
                selp.f32 %f4, %f{10 + i}, %f4, %p10;
                """);
        }

        b.AppendLine(BlockReduce("RTK", "%f4", "max", NegInf));
        b.AppendLine("""
            mov.f32 %f3, %f4;
            add.u32 %r12, %r12, 1;
            bra TK;
            TK_END:
            setp.eq.u32 %p13, 0, 0;
            COUNT:
            mov.u32 %r13, 0;
            """);
        // Kept: at or above the threshold (%p13 set), or strictly above it (the retry after ties overflow the slots).
        string Kept(int i) => $"""
            setp.ge.and.f32 %p10, %f{10 + i}, %f3, %p13;
            setp.gt.or.f32 %p10, %f{10 + i}, %f3, %p10;
            setp.gt.and.f32 %p10, %f{10 + i}, {NegInf}, %p10;
            """;
        for (int i = 0; i < 8; i++)
        {
            b.AppendLine(Kept(i));
            b.AppendLine("selp.u32 %r14, 1, 0, %p10;");
            b.AppendLine("add.u32 %r13, %r13, %r14;");
        }

        // Exclusive prefix of the per-thread counts (thread 0, in order), so candidates keep vocabulary order.
        b.AppendLine($"""
            shl.b32 %r15, %tx, 2;
            add.u32 %r15, %r15, %spart;
            st.shared.u32 [%r15], %r13;
            bar.sync 0;
            setp.ne.u32 %p11, %tx, 0;
            @%p11 bra SCAN_DONE;
            mov.u32 %r16, 0;
            mov.u32 %r17, 0;
            SC:
            setp.ge.u32 %p12, %r17, %nt;
            @%p12 bra SC_END;
            shl.b32 %r18, %r17, 2;
            add.u32 %r18, %r18, %spart;
            ld.shared.u32 %r19, [%r18];
            st.shared.u32 [%r18], %r16;
            add.u32 %r16, %r16, %r19;
            add.u32 %r17, %r17, 1;
            bra SC;
            SC_END:
            st.shared.u32 [%sb], %r16;
            SCAN_DONE:
            bar.sync 0;
            ld.shared.u32 %r20, [%r15];
            ld.shared.u32 %r21, [%sb];
            bar.sync 0;
            setp.gt.u32 %p15, %r21, {CandidateSlots};
            and.pred %p15, %p15, %p13;
            @!%p15 bra COUNTED;
            setp.ne.u32 %p13, 0, 0;
            bra COUNT;
            COUNTED:
            shl.b32 %r22, %row, {(int)Math.Log2(CandidateSlots)};
            mul.wide.u32 %rd4, %r22, 4;
            add.u64 %rd5, %b_candv, %rd4;
            add.u64 %rd6, %b_candi, %rd4;
            """);
        for (int i = 0; i < 8; i++)
        {
            b.AppendLine(Kept(i));
            b.AppendLine($"""
                setp.lt.u32 %p11, %r20, {CandidateSlots};
                and.pred %p12, %p10, %p11;
                mul.wide.u32 %rd7, %r20, 4;
                add.u64 %rd8, %rd7, %rd5;
                @%p12 st.global.f32 [%rd8], %f{10 + i};
                add.u32 %r23, %r10, {i};
                cvt.rn.f32.u32 %f5, %r23;
                add.u64 %rd9, %rd7, %rd6;
                @%p12 st.global.f32 [%rd9], %f5;
                selp.u32 %r14, 1, 0, %p10;
                add.u32 %r20, %r20, %r14;
                """);
        }

        b.AppendLine($"""
            setp.lt.u32 %p10, %tx, {CandidateSlots};
            setp.ge.and.u32 %p10, %tx, %r21, %p10;
            mul.wide.u32 %rd7, %tx, 4;
            add.u64 %rd8, %rd7, %rd5;
            @%p10 st.global.f32 [%rd8], {NegInf};
            add.u64 %rd9, %rd7, %rd6;
            @%p10 st.global.f32 [%rd9], {F(-1f)};
            setp.ne.u32 %p10, %tx, 0;
            @%p10 bra FLAGS_DONE;
            mul.wide.u32 %rd7, %r7, 4;
            add.u64 %rd7, %rd7, %b_flags;
            setp.gt.u32 %p10, %r21, {CandidateSlots};
            @!%p10 bra STRICT_FLAG;
            mov.u32 %r24, 0xFFFFFFFF;
            st.global.u32 [%rd7], %r24;
            bra FLAGS_DONE;
            STRICT_FLAG:
            @%p13 bra FLAGS_DONE;
            mov.b32 %r24, %f3;
            shr.s32 %r25, %r24, 31;
            or.b32 %r25, %r25, 0x80000000;
            xor.b32 %r24, %r24, %r25;
            atom.global.max.u32 %r26, [%rd7], %r24;
            FLAGS_DONE:
            """);
        RowBlock(sb, "topk_candidates_f32", ["logits", "candv", "candi", "flags"],
            [("u32", "vocab"), ("u32", "rowstride"), ("u32", "rowoffset"), ("f32", "invt"), ("u32", "topk"), ("u32", "bpr")],
            b.ToString(), sharedFloats: RowThreads);
    }

    private static void PenaltyKernels(StringBuilder sb)
    {
        // Repetition penalties, one block per row: copy the row to work[r, :]; then each thread takes history entries
        // k (newest first) of the last n and, when k is the token's most recent occurrence, penalizes it once using its
        // count in the window (as CpuBackend.PenalizeRows).
        RowBlock(sb, "penalize_rows_f32", ["logits", "work", "history", "len"],
            [("u32", "vocab"), ("u32", "rowstride"), ("u32", "rowoffset"), ("u32", "cap"), ("u32", "lastn"), ("f32", "repeat"), ("f32", "presence"), ("f32", "frequency"), ("u32", "rows")],
            $"""
            mad.lo.u32 %r5, %row, %s_rowstride, %s_rowoffset;
            mul.wide.u32 %rd1, %r5, 4;
            add.u64 %rd2, %b_logits, %rd1;
            mul.lo.u32 %r6, %row, %s_vocab;
            mul.wide.u32 %rd1, %r6, 4;
            add.u64 %rd3, %b_work, %rd1;
            sub.u64 %rd6, %rd3, %rd2;
            """ + "\n" + StridedLoop("PC", "%rd2", "%s_vocab", """
            add.u64 %rd7, %rd5, %rd6;
            st.global.f32 [%rd7], %f2;
            """) + "\n" + $"""
            bar.sync 0;
            ld.global.f32 %f1, [%b_len];
            cvt.rzi.u32.f32 %r7, %f1;
            min.u32 %r8, %s_lastn, %s_cap;
            min.u32 %r8, %r8, %r7;
            mul.lo.u32 %r9, %row, %s_cap;
            mul.wide.u32 %rd1, %r9, 4;
            add.u64 %rd4, %b_history, %rd1;
            mov.u32 %r10, %tx;
            PK:
            setp.ge.u32 %p1, %r10, %r8;
            @%p1 bra PK_END;
            sub.u32 %r11, %r7, 1;
            sub.u32 %r11, %r11, %r10;
            rem.u32 %r11, %r11, %s_cap;
            mul.wide.u32 %rd8, %r11, 4;
            add.u64 %rd8, %rd8, %rd4;
            ld.global.f32 %f2, [%rd8];
            cvt.rzi.u32.f32 %r12, %f2;
            mov.u32 %r13, 0;
            mov.u32 %r14, 0;
            mov.u32 %r15, 0;
            PQ:
            setp.ge.u32 %p2, %r15, %r8;
            @%p2 bra PQ_END;
            sub.u32 %r16, %r7, 1;
            sub.u32 %r16, %r16, %r15;
            rem.u32 %r16, %r16, %s_cap;
            mul.wide.u32 %rd9, %r16, 4;
            add.u64 %rd9, %rd9, %rd4;
            ld.global.f32 %f3, [%rd9];
            cvt.rzi.u32.f32 %r17, %f3;
            setp.eq.u32 %p3, %r17, %r12;
            selp.u32 %r18, 1, 0, %p3;
            add.u32 %r13, %r13, %r18;
            setp.lt.u32 %p4, %r15, %r10;
            and.pred %p4, %p4, %p3;
            selp.u32 %r18, 1, 0, %p4;
            or.b32 %r14, %r14, %r18;
            add.u32 %r15, %r15, 1;
            bra PQ;
            PQ_END:
            setp.ne.u32 %p5, %r14, 0;
            setp.ge.or.u32 %p5, %r12, %s_vocab, %p5;
            @%p5 bra PK_NEXT;
            mul.wide.u32 %rd10, %r12, 4;
            add.u64 %rd10, %rd10, %rd3;
            ld.global.f32 %f5, [%rd10];
            setp.gt.f32 %p6, %f5, {Zero};
            div.rn.f32 %f6, %f5, %s_repeat;
            mul.f32 %f7, %f5, %s_repeat;
            selp.f32 %f5, %f6, %f7, %p6;
            sub.f32 %f5, %f5, %s_presence;
            cvt.rn.f32.u32 %f8, %r13;
            mul.f32 %f8, %f8, %s_frequency;
            sub.f32 %f5, %f5, %f8;
            st.global.f32 [%rd10], %f5;
            PK_NEXT:
            add.u32 %r10, %r10, %nt;
            bra PK;
            PK_END:
            """);

        // history[r, len % cap] = ids[r]
        Elementwise(sb, "history_push_f32", ["ids", "history", "len"], [("u32", "cap"), ("u32", "rows")],
            """
            ld.global.f32 %f1, [%b_len];
            cvt.rzi.u32.f32 %r5, %f1;
            rem.u32 %r5, %r5, %s_cap;
            mad.lo.u32 %r5, %i, %s_cap, %r5;
            mul.wide.u32 %rd1, %r5, 4;
            add.u64 %rd1, %b_history, %rd1;
            ld.global.f32 %f2, [%a_ids];
            st.global.f32 [%rd1], %f2;
            """);
    }
}
