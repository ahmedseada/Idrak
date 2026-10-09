// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;

namespace Idrak.Gpu.Cuda;

// Attention over one range of keys per query row (Backend.AttentionSpans), float32, tiled as attention_flash_f32.
internal static partial class PtxKernels
{
    /// <summary>The span attention kernel's name (forward; the gradient: PtxKernels.SpansBackward.cs).</summary>
    public const string SpanAttentionName = "attention_spans_f32";

    // o[h, i] = Σ_c softmax(cap(scale · q[h, i] · k[g, c])) · v[g, c] over st[t·rows + i] ≤ c < en[t·rows + i] (clamped to
    // [0, keyRows]), g = h / group, t = h / headsPerTable; q, o [heads, rows, dim], k, v [kvHeads, keyRows, dim]. As
    // attention_flash_f32: a block of 4 warps takes 32 query rows (8 per warp) of one head and walks the keys in tiles of
    // 32 positions (keys transposed in shared memory); lane p scores position p against the warp's 8 rows, the running max
    // and sum update per row (online softmax), and the lanes then split the head dimension to add Σ_p weight_p · v_p. The
    // tiles go from the smallest start of the block's rows to their largest end (lane l of every warp reads row l's range
    // and the warp reduces them, so every warp walks the same tiles); a position outside a row's range scores -∞ for it.
    // A row whose range is empty keeps a sum of 0: zeros, and a log-sum-exp of -∞. Grid: x = ⌈rows / 32⌉, y = heads.
    private static void AttentionSpans(StringBuilder sb)
    {
        const int T = FlashTile, D = FlashMaxDim, Rows = 8;
        var s = new StringBuilder();
        s.AppendLine($$"""
            .visible .entry {{SpanAttentionName}}(
                .param .u64 p_q, .param .u64 p_k, .param .u64 p_v, .param .u64 p_st, .param .u64 p_en, .param .u64 p_o, .param .u64 p_lse,
                .param .u32 p_rows, .param .u32 p_keys, .param .u32 p_dim, .param .f32 p_scale, .param .u32 p_group, .param .u32 p_hpt,
                .param .f32 p_softcap
            )
            {
                .reg .pred %p<16>;
                .reg .b32 %lo<{{Rows}}>;
                .reg .b32 %hi<{{Rows}}>;
                .reg .b32 %from, %to, %tb, %bs, %be;
                .reg .f32 %softcap, %captmp;
                .reg .pred %capped;
                .reg .f32 %o<{{Rows * 4}}>;
                .reg .f32 %m<{{Rows}}>;
                .reg .f32 %l<{{Rows}}>;
                .reg .f32 %s<{{Rows}}>;
                .reg .f32 %f<16>;
                .reg .b32 %r<48>;
                .reg .b64 %rd<24>;
                .reg .pred %dv<4>;
                .shared .align 16 .f32 sa_q[{{T * D}}];
                .shared .align 16 .f32 sa_k[{{T * D}}];
                .shared .align 16 .f32 sa_v[{{T * D}}];
                ld.param.u64 %rd1, [p_q];
                ld.param.u64 %rd2, [p_k];
                ld.param.u64 %rd3, [p_v];
                ld.param.u64 %rd4, [p_st];
                ld.param.u64 %rd20, [p_en];
                ld.param.u64 %rd5, [p_o];
                ld.param.u64 %rd6, [p_lse];
                cvta.to.global.u64 %rd1, %rd1;
                cvta.to.global.u64 %rd2, %rd2;
                cvta.to.global.u64 %rd3, %rd3;
                cvta.to.global.u64 %rd4, %rd4;
                cvta.to.global.u64 %rd20, %rd20;
                cvta.to.global.u64 %rd5, %rd5;
                setp.ne.u64 %p15, %rd6, 0;
                @%p15 cvta.to.global.u64 %rd6, %rd6;
                ld.param.u32 %r1, [p_rows];
                ld.param.u32 %r3, [p_keys];
                ld.param.u32 %r4, [p_dim];
                ld.param.f32 %f15, [p_scale];
                ld.param.f32 %softcap, [p_softcap];
                setp.gt.f32 %capped, %softcap, 0f00000000;
                mov.u32 %r6, %tid.x;
                and.b32 %r7, %r6, 31;
                shr.u32 %r8, %r6, 5;
                mov.u32 %r9, %ctaid.x;
                shl.b32 %r9, %r9, {{(int)Math.Log2(T)}};
                mov.u32 %r10, %ctaid.y;
                mul.lo.u32 %r11, %r10, %r1;
                ld.param.u32 %r2, [p_group];
                div.u32 %r12, %r10, %r2;
                mul.lo.u32 %r12, %r12, %r3;
                ld.param.u32 %r2, [p_hpt];
                div.u32 %tb, %r10, %r2;
                mul.lo.u32 %tb, %tb, %r1;
                cvt.u64.u32 %rd7, %r11;
                cvt.u64.u32 %rd8, %r4;
                mul.lo.u64 %rd9, %rd7, %rd8;
                shl.b64 %rd9, %rd9, 2;
                add.u64 %rd10, %rd1, %rd9;
                add.u64 %rd11, %rd5, %rd9;
                cvt.u64.u32 %rd7, %r12;
                mul.lo.u64 %rd12, %rd7, %rd8;
                shl.b64 %rd12, %rd12, 2;
                add.u64 %rd13, %rd2, %rd12;
                add.u64 %rd14, %rd3, %rd12;
                mul.lo.u32 %r13, %r4, {{T}};
            """);

        // Query tile: sa_q[r, d] for rows row0 + r (zeros past the head's rows).
        s.AppendLine($$"""
                mov.u32 %r14, %r6;
            QL:
                setp.ge.u32 %p1, %r14, %r13;
                @%p1 bra QL_END;
                div.u32 %r15, %r14, %r4;
                rem.u32 %r16, %r14, %r4;
                add.u32 %r17, %r9, %r15;
                setp.lt.u32 %p2, %r17, %r1;
                mad.lo.u32 %r18, %r17, %r4, %r16;
                mul.wide.u32 %rd15, %r18, 4;
                add.u64 %rd15, %rd15, %rd10;
                mov.f32 %f2, 0f00000000;
                @%p2 ld.global.f32 %f2, [%rd15];
                mad.lo.u32 %r19, %r15, {{D}}, %r16;
                shl.b32 %r19, %r19, 2;
                mov.u32 %r20, sa_q;
                add.u32 %r19, %r19, %r20;
                st.shared.f32 [%r19], %f2;
                add.u32 %r14, %r14, 128;
                bra QL;
            QL_END:
            """);

        // %bs, %be = the range of the table row `row` (a register), clamped to [0, keyRows]; 0, 0 past the head's rows.
        static string Range(string row, string pred) => $$"""
                mov.u32 %bs, 0;
                mov.u32 %be, 0;
                setp.lt.u32 {{pred}}, {{row}}, %r1;
                add.u32 %r22, %tb, {{row}};
                mul.wide.u32 %rd15, %r22, 4;
                add.u64 %rd16, %rd15, %rd4;
                add.u64 %rd17, %rd15, %rd20;
                @{{pred}} ld.global.f32 %f2, [%rd16];
                @{{pred}} ld.global.f32 %f3, [%rd17];
                @{{pred}} cvt.rzi.s32.f32 %bs, %f2;
                @{{pred}} cvt.rzi.s32.f32 %be, %f3;
                max.s32 %bs, %bs, 0;
                min.s32 %bs, %bs, %r3;
                max.s32 %be, %be, 0;
                min.s32 %be, %be, %r3;
            """;

        // Each row's range, its running max, sum and outputs.
        for (int i = 0; i < Rows; i++)
        {
            s.AppendLine($$"""
                    mad.lo.u32 %r21, %r8, {{Rows}}, {{i}};
                    add.u32 %r21, %r21, %r9;
                    {{Range("%r21", "%p2")}}
                    mov.u32 %lo{{i}}, %bs;
                    mov.u32 %hi{{i}}, %be;
                    mov.f32 %m{{i}}, 0fFF800000;
                    mov.f32 %l{{i}}, 0f00000000;
                """);
            for (int j = 0; j < 4; j++)
            {
                s.AppendLine($"    mov.f32 %o{i * 4 + j}, 0f00000000;");
            }
        }

        // The block's keys: lane l reads row row0 + l's range; the warp takes the smallest start and largest end of the
        // non-empty ones (an empty range counts as [keyRows, 0)).
        s.AppendLine($$"""
                add.u32 %r21, %r9, %r7;
                {{Range("%r21", "%p2")}}
                setp.gt.u32 %p3, %be, %bs;
                selp.b32 %from, %bs, %r3, %p3;
                selp.b32 %to, %be, 0, %p3;
            """);
        foreach (int offset in new[] { 16, 8, 4, 2, 1 })
        {
            s.AppendLine($"    shfl.sync.bfly.b32 %r22, %from, {offset}, 31, 0xffffffff;");
            s.AppendLine("    min.u32 %from, %from, %r22;");
            s.AppendLine($"    shfl.sync.bfly.b32 %r22, %to, {offset}, 31, 0xffffffff;");
            s.AppendLine("    max.u32 %to, %to, %r22;");
        }

        for (int j = 0; j < 4; j++)
        {
            s.AppendLine($"    add.u32 %r38, %r7, {32 * j};");
            s.AppendLine($"    setp.lt.u32 %dv{j}, %r38, %r4;");
        }

        s.AppendLine($$"""
                mov.u32 %r39, %from;
            TILE:
                setp.ge.u32 %p4, %r39, %to;
                @%p4 bra TILE_END;
                bar.sync 0;
                mov.u32 %r14, %r6;
            KL:
                setp.ge.u32 %p1, %r14, %r13;
                @%p1 bra KL_END;
                div.u32 %r15, %r14, %r4;
                rem.u32 %r16, %r14, %r4;
                add.u32 %r17, %r39, %r15;
                setp.lt.u32 %p2, %r17, %to;
                mov.f32 %f2, 0f00000000;
                mov.f32 %f3, 0f00000000;
                mad.lo.u32 %r18, %r17, %r4, %r16;
                mul.wide.u32 %rd15, %r18, 4;
                add.u64 %rd16, %rd15, %rd13;
                add.u64 %rd17, %rd15, %rd14;
                @%p2 ld.global.f32 %f2, [%rd16];
                @%p2 ld.global.f32 %f3, [%rd17];
                mad.lo.u32 %r19, %r16, {{T}}, %r15;
                shl.b32 %r19, %r19, 2;
                mov.u32 %r20, sa_k;
                add.u32 %r19, %r19, %r20;
                st.shared.f32 [%r19], %f2;
                mad.lo.u32 %r19, %r15, {{D}}, %r16;
                shl.b32 %r19, %r19, 2;
                mov.u32 %r20, sa_v;
                add.u32 %r19, %r19, %r20;
                st.shared.f32 [%r19], %f3;
                add.u32 %r14, %r14, 128;
                bra KL;
            KL_END:
                bar.sync 0;
            """);
        for (int i = 0; i < Rows; i++)
        {
            s.AppendLine($"    mov.f32 %s{i}, 0f00000000;");
        }

        // Scores: lane = position; sa_k[d, lane], sa_q[warp·8 + i, d] (broadcast).
        s.AppendLine($$"""
                mov.u32 %r40, sa_k;
                shl.b32 %r41, %r7, 2;
                add.u32 %r40, %r40, %r41;
                mov.u32 %r42, sa_q;
                mul.lo.u32 %r43, %r8, {{Rows * D * 4}};
                add.u32 %r42, %r42, %r43;
                mov.u32 %r44, 0;
            DOT:
                setp.ge.u32 %p5, %r44, %r4;
                @%p5 bra DOT_END;
                ld.shared.f32 %f4, [%r40];
            """);
        for (int i = 0; i < Rows; i++)
        {
            s.AppendLine($"    ld.shared.f32 %f5, [%r42+{i * D * 4}];");
            s.AppendLine($"    fma.rn.f32 %s{i}, %f5, %f4, %s{i};");
        }

        s.AppendLine($$"""
                add.u32 %r40, %r40, {{T * 4}};
                add.u32 %r42, %r42, 4;
                add.u32 %r44, %r44, 1;
                bra DOT;
            DOT_END:
                add.u32 %r45, %r39, %r7;
                mov.u32 %r46, sa_v;
                shl.b32 %r47, %r7, 2;
                add.u32 %r46, %r46, %r47;
            """);
        for (int i = 0; i < Rows; i++)
        {
            // Mask by the row's range, then the online softmax update for row i and its weighted values.
            s.AppendLine($$"""
                    setp.lt.u32 %p6, %r45, %hi{{i}};
                    setp.ge.and.u32 %p6, %r45, %lo{{i}}, %p6;
                    mul.f32 %s{{i}}, %s{{i}}, %f15;
                    {{SoftcapPtx($"%s{i}", "%softcap", "%capped", "%captmp")}}
                    selp.f32 %s{{i}}, %s{{i}}, 0fFF800000, %p6;
                    mov.f32 %f6, %s{{i}};
                """);
            foreach (int offset in new[] { 16, 8, 4, 2, 1 })
            {
                s.AppendLine($"    shfl.sync.bfly.b32 %f7, %f6, {offset}, 31, 0xffffffff;");
                s.AppendLine("    max.f32 %f6, %f6, %f7;");
            }

            s.AppendLine($$"""
                    max.f32 %f8, %m{{i}}, %f6;
                    setp.eq.f32 %p7, %f8, 0fFF800000;
                    selp.f32 %f9, 0f00000000, %f8, %p7;
                    sub.f32 %f10, %m{{i}}, %f9;
                    mul.f32 %f10, %f10, 0f3FB8AA3B;
                    ex2.approx.ftz.f32 %f10, %f10;
                    @%p7 mov.f32 %f10, 0f3F800000;
                    sub.f32 %f11, %s{{i}}, %f9;
                    mul.f32 %f11, %f11, 0f3FB8AA3B;
                    ex2.approx.ftz.f32 %f11, %f11;
                    mov.f32 %f12, %f11;
                """);
            foreach (int offset in new[] { 16, 8, 4, 2, 1 })
            {
                s.AppendLine($"    shfl.sync.bfly.b32 %f7, %f12, {offset}, 31, 0xffffffff;");
                s.AppendLine("    add.f32 %f12, %f12, %f7;");
            }

            s.AppendLine($"    fma.rn.f32 %l{i}, %l{i}, %f10, %f12;");
            s.AppendLine($"    mov.f32 %m{i}, %f8;");
            for (int j = 0; j < 4; j++)
            {
                s.AppendLine($"    mul.f32 %o{i * 4 + j}, %o{i * 4 + j}, %f10;");
            }

            s.AppendLine("    mov.u32 %r47, %r46;");
            s.AppendLine("    mov.u32 %r20, 0;");
            s.AppendLine($"SPV{i}:");
            s.AppendLine("    setp.ge.u32 %p8, %r20, 32;");
            s.AppendLine($"    @%p8 bra SPV{i}_END;");
            s.AppendLine("    shfl.sync.idx.b32 %f13, %f11, %r20, 31, 0xffffffff;");
            for (int j = 0; j < 4; j++)
            {
                s.AppendLine($"    @%dv{j} ld.shared.f32 %f14, [%r47+{128 * j}];");
                s.AppendLine($"    @%dv{j} fma.rn.f32 %o{i * 4 + j}, %f13, %f14, %o{i * 4 + j};");
            }

            s.AppendLine($"    add.u32 %r47, %r47, {D * 4};");
            s.AppendLine("    add.u32 %r20, %r20, 1;");
            s.AppendLine($"    bra SPV{i};");
            s.AppendLine($"SPV{i}_END:");
        }

        s.AppendLine($$"""
                add.u32 %r39, %r39, {{T}};
                bra TILE;
            TILE_END:
            """);
        for (int i = 0; i < Rows; i++)
        {
            // o /= l (zeros when the row saw nothing); lse = m + ln l (-∞ then). Rows past the head's rows are not written.
            s.AppendLine($$"""
                    mad.lo.u32 %r21, %r8, {{Rows}}, {{i}};
                    add.u32 %r21, %r21, %r9;
                    setp.ge.u32 %p9, %r21, %r1;
                    @%p9 bra SWRITTEN{{i}};
                    setp.gt.f32 %p11, %l{{i}}, 0f00000000;
                    rcp.rn.f32 %f13, %l{{i}};
                    selp.f32 %f13, %f13, 0f00000000, %p11;
                    mul.lo.u32 %r22, %r21, %r4;
                    add.u32 %r22, %r22, %r7;
                    mul.wide.u32 %rd18, %r22, 4;
                    add.u64 %rd18, %rd18, %rd11;
                """);
            for (int j = 0; j < 4; j++)
            {
                s.AppendLine($"    mul.f32 %f14, %o{i * 4 + j}, %f13;");
                s.AppendLine($"    @%dv{j} st.global.f32 [%rd18+{128 * j}], %f14;");
            }

            s.AppendLine($$"""
                    setp.eq.u32 %p10, %r7, 0;
                    and.pred %p10, %p10, %p15;
                    @!%p10 bra SWRITTEN{{i}};
                    lg2.approx.ftz.f32 %f14, %l{{i}};
                    fma.rn.f32 %f14, %f14, 0f3F317218, %m{{i}};
                    selp.f32 %f14, %f14, 0fFF800000, %p11;
                    add.u32 %r23, %r11, %r21;
                    mul.wide.u32 %rd19, %r23, 4;
                    add.u64 %rd19, %rd19, %rd6;
                    st.global.f32 [%rd19], %f14;
                SWRITTEN{{i}}:
                """);
        }

        s.AppendLine("""
                ret;
            }
            """);
        sb.AppendLine(s.ToString());
    }
}
