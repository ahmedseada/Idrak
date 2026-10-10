// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;

namespace Idrak.Gpu.Cuda;

// The gradient of attention over one range of keys per query row (Backend.AttentionSpansBackward), float32, built as
// the backward of attention_flash_f32 (attn_bwd_kv_f32, attn_bwd_q_f32) with each row's range read from the starts and
// ends buffers instead of a causal limit and a window, and grouped heads: the weights are recomputed from each row's
// log-sum-exp, P = exp(cap(scale · q·k) - lse), dS = P ∘ (dO·v - Δ) · cap' with Δ = dO · O per row (attn_bwd_d_f32),
// then dV += Pᵀ dO, dK += scale · dSᵀ Q, dQ += scale · dS K. Two kernels, no atomics (deterministic: every gradient
// element has one owner that adds its terms in a fixed order):
//   attn_spans_bwd_kv_f32  a block of 4 warps owns 16 keys (4 per warp) of one key/value head and walks every query
//                          head of its group, and each head's query rows in tiles of 32 held in shared memory (rows
//                          padded to 129 floats); a tile none of whose rows sees a key of the block is skipped (the
//                          lanes read the tile's 32 ranges and vote); grid x = ⌈keyRows / 16⌉, y = kvHeads;
//   attn_spans_bwd_q_f32   a block owns 32 query rows (8 per warp) of one head and walks the keys from the smallest
//                          start of its rows to their largest end in tiles of 32 held transposed in shared memory
//                          (padded to 33); grid x = ⌈rows / 32⌉, y = heads.
// Head sizes up to FlashMaxDim (4 dimensions per lane). Rows whose range is empty add nothing.
internal static partial class PtxKernels
{
    /// <summary>The span attention gradient's kernels: dK and dV, then dQ (the per-row Δ is attn_bwd_d_f32's).</summary>
    public const string SpanBackwardKeysName = "attn_spans_bwd_kv_f32", SpanBackwardQueriesName = "attn_spans_bwd_q_f32";

    /// <summary>Keys per block of <see cref="SpanBackwardKeysName"/>, and query rows per block of <see cref="SpanBackwardQueriesName"/>.</summary>
    public const int SpanBackwardKeys = 16, SpanBackwardRows = 32;

    // The parameters of both kernels (one argument list serves the two launches).
    private const string SpanBackwardParameters = """
            .param .u64 p_q, .param .u64 p_k, .param .u64 p_v, .param .u64 p_st, .param .u64 p_en, .param .u64 p_dout,
            .param .u64 p_lse, .param .u64 p_dd, .param .u64 p_dq, .param .u64 p_dk, .param .u64 p_dv,
            .param .u32 p_rows, .param .u32 p_keys, .param .u32 p_dim, .param .f32 p_scale, .param .u32 p_group, .param .u32 p_hpt,
            .param .f32 p_softcap
        """;

    // Common prologue: %r1 rows, %r3 keyRows, %r4 dim, %f31 scale, %f30 the soft-cap (%p15 when there is one), %r50 the
    // query heads per key/value head, %r51 the heads per table, %r6 tid, %r7 lane, %r8 warp; the bases of q (%rd31), dO
    // (%rd32), lse (%rd33), Δ (%rd34), dq (%rd35), the starts (%rd25) and ends (%rd26); k, v, dk, dv in %rd2, %rd3, %rd8,
    // %rd9 (moved to a key/value head by SpanKvPointers); %rd30 = dim.
    private const string SpanBackwardPrologue = """
            ld.param.u32 %r1, [p_rows];
            ld.param.u32 %r3, [p_keys];
            ld.param.u32 %r4, [p_dim];
            ld.param.f32 %f31, [p_scale];
            ld.param.f32 %f30, [p_softcap];
            setp.gt.f32 %p15, %f30, 0f00000000;
            ld.param.u32 %r50, [p_group];
            ld.param.u32 %r51, [p_hpt];
            mov.u32 %r6, %tid.x;
            and.b32 %r7, %r6, 31;
            shr.u32 %r8, %r6, 5;
            cvt.u64.u32 %rd30, %r4;
            ld.param.u64 %rd31, [p_q];
            cvta.to.global.u64 %rd31, %rd31;
            ld.param.u64 %rd32, [p_dout];
            cvta.to.global.u64 %rd32, %rd32;
            ld.param.u64 %rd33, [p_lse];
            cvta.to.global.u64 %rd33, %rd33;
            ld.param.u64 %rd34, [p_dd];
            cvta.to.global.u64 %rd34, %rd34;
            ld.param.u64 %rd35, [p_dq];
            cvta.to.global.u64 %rd35, %rd35;
            ld.param.u64 %rd25, [p_st];
            cvta.to.global.u64 %rd25, %rd25;
            ld.param.u64 %rd26, [p_en];
            cvta.to.global.u64 %rd26, %rd26;
            ld.param.u64 %rd2, [p_k];
            cvta.to.global.u64 %rd2, %rd2;
            ld.param.u64 %rd3, [p_v];
            cvta.to.global.u64 %rd3, %rd3;
            ld.param.u64 %rd8, [p_dk];
            cvta.to.global.u64 %rd8, %rd8;
            ld.param.u64 %rd9, [p_dv];
            cvta.to.global.u64 %rd9, %rd9;
        """;

    // Pointers of query head `h` (a register): q %rd1, dO %rd4, dq %rd7 ([heads, rows, dim]), lse %rd5, Δ %rd6
    // ([heads, rows]), and %tb = the first entry of its table of ranges. Uses %r52, %r53, %rd28, %rd29.
    private static string SpanHeadPointers(string h) => $"""
            mul.lo.u32 %r52, {h}, %r1;
            mul.wide.u32 %rd28, %r52, 4;
            add.u64 %rd5, %rd33, %rd28;
            add.u64 %rd6, %rd34, %rd28;
            cvt.u64.u32 %rd29, %r52;
            mul.lo.u64 %rd28, %rd29, %rd30;
            shl.b64 %rd28, %rd28, 2;
            add.u64 %rd1, %rd31, %rd28;
            add.u64 %rd4, %rd32, %rd28;
            add.u64 %rd7, %rd35, %rd28;
            div.u32 %r53, {h}, %r51;
            mul.lo.u32 %tb, %r53, %r1;
        """;

    // Moves k, v, dk, dv to key/value head `g` (a register; once per kernel). Uses %r52, %rd28, %rd29.
    private static string SpanKvPointers(string g) => $"""
            mul.lo.u32 %r52, {g}, %r3;
            cvt.u64.u32 %rd29, %r52;
            mul.lo.u64 %rd28, %rd29, %rd30;
            shl.b64 %rd28, %rd28, 2;
            add.u64 %rd2, %rd2, %rd28;
            add.u64 %rd3, %rd3, %rd28;
            add.u64 %rd8, %rd8, %rd28;
            add.u64 %rd9, %rd9, %rd28;
        """;

    // `lo`, `hi` = the range of table row `row` (a register), clamped to [0, keyRows]; 0, 0 past the head's rows (`valid`
    // false then). Uses %r54, %rd27, %rd36, %rd37, %f28, %f29.
    private static string SpanRowRange(string row, string lo, string hi, string valid) => $"""
            mov.u32 {lo}, 0;
            mov.u32 {hi}, 0;
            setp.lt.u32 {valid}, {row}, %r1;
            add.u32 %r54, %tb, {row};
            mul.wide.u32 %rd27, %r54, 4;
            add.u64 %rd36, %rd27, %rd25;
            add.u64 %rd37, %rd27, %rd26;
            @{valid} ld.global.f32 %f28, [%rd36];
            @{valid} ld.global.f32 %f29, [%rd37];
            @{valid} cvt.rzi.s32.f32 {lo}, %f28;
            @{valid} cvt.rzi.s32.f32 {hi}, %f29;
            max.s32 {lo}, {lo}, 0;
            min.s32 {lo}, {lo}, %r3;
            max.s32 {hi}, {hi}, 0;
            min.s32 {hi}, {hi}, %r3;
        """;

    private static void AttentionSpansBackward(StringBuilder sb)
    {
        AttentionSpansBackwardKeys(sb);
        AttentionSpansBackwardQueries(sb);
    }

    private static void AttentionSpansBackwardKeys(StringBuilder sb)
    {
        const int Keys = 4, Stride = 129, QT = 32;
        var s = new StringBuilder();
        s.AppendLine(CultureInfo.InvariantCulture, $$"""
            .visible .entry {{SpanBackwardKeysName}}(
            {{SpanBackwardParameters}}
            )
            {
                .reg .pred %p<16>;
                .reg .pred %dv<4>;
                .reg .f32 %gk<{{Keys * 4}}>;
                .reg .f32 %gv<{{Keys * 4}}>;
                .reg .f32 %sp<{{Keys}}>;
                .reg .f32 %sd<{{Keys}}>;
                .reg .f32 %f<32>;
                .reg .b32 %r<64>;
                .reg .b64 %rd<40>;
                .reg .b32 %lo, %hi, %h, %hend, %tb, %ke;
                .shared .align 4 .f32 skv_q[{{QT * Stride}}];
                .shared .align 4 .f32 skv_do[{{QT * Stride}}];
            {{SpanBackwardPrologue}}
                mov.u32 %r13, %ctaid.x;
                shl.b32 %r13, %r13, 4;
                shl.b32 %r14, %r8, 2;
                add.u32 %r14, %r14, %r13;
                add.u32 %ke, %r13, {{SpanBackwardKeys}};
                min.u32 %ke, %ke, %r3;
                mul.lo.u32 %r15, %r4, {{QT}};
                mov.u32 %r16, skv_q;
                mov.u32 %r17, skv_do;
                mov.u32 %r55, %ctaid.y;
            {{SpanKvPointers("%r55")}}
                mul.lo.u32 %h, %r55, %r50;
                add.u32 %hend, %h, %r50;
            """);
        for (int j = 0; j < 4; j++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    add.u32 %r18, %r7, {32 * j};");
            s.AppendLine(CultureInfo.InvariantCulture, $"    setp.lt.u32 %dv{j}, %r18, %r4;");
        }

        for (int i = 0; i < Keys * 4; i++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    mov.f32 %gk{i}, 0f00000000;");
            s.AppendLine(CultureInfo.InvariantCulture, $"    mov.f32 %gv{i}, 0f00000000;");
        }

        // Every query head of the group, its rows a tile at a time. Lane l takes row r0 + l's range; a tile is skipped
        // when no lane's range meets the block's keys [%r13, %ke) (every warp reads the same 32 ranges: the same vote).
        s.AppendLine(CultureInfo.InvariantCulture, $$"""
            HL:
                setp.ge.u32 %p1, %h, %hend;
                @%p1 bra HL_END;
            {{SpanHeadPointers("%h")}}
                mov.u32 %r19, 0;
            QT:
                setp.ge.u32 %p1, %r19, %r1;
                @%p1 bra QT_END;
                add.u32 %r32, %r19, %r7;
            {{SpanRowRange("%r32", "%lo", "%hi", "%p5")}}
                setp.lt.u32 %p2, %lo, %ke;
                setp.gt.and.u32 %p2, %hi, %r13, %p2;
                setp.gt.and.u32 %p2, %hi, %lo, %p2;
                vote.sync.any.pred %p2, %p2, 0xffffffff;
                @!%p2 bra QT_NEXT;
                bar.sync 0;
                mov.u32 %r25, %r6;
            QL:
                setp.ge.u32 %p3, %r25, %r15;
                @%p3 bra QL_END;
                div.u32 %r26, %r25, %r4;
                rem.u32 %r27, %r25, %r4;
                add.u32 %r28, %r19, %r26;
                setp.lt.u32 %p4, %r28, %r1;
                mad.lo.u32 %r29, %r28, %r4, %r27;
                mul.wide.u32 %rd10, %r29, 4;
                add.u64 %rd11, %rd10, %rd1;
                add.u64 %rd12, %rd10, %rd4;
                mov.f32 %f1, 0f00000000;
                mov.f32 %f2, 0f00000000;
                @%p4 ld.global.f32 %f1, [%rd11];
                @%p4 ld.global.f32 %f2, [%rd12];
                mad.lo.u32 %r30, %r26, {{Stride}}, %r27;
                shl.b32 %r30, %r30, 2;
                add.u32 %r31, %r30, %r16;
                st.shared.f32 [%r31], %f1;
                add.u32 %r31, %r30, %r17;
                st.shared.f32 [%r31], %f2;
                add.u32 %r25, %r25, 128;
                bra QL;
            QL_END:
                bar.sync 0;
                mul.wide.u32 %rd13, %r32, 4;
                mov.f32 %f3, 0f00000000;
                mov.f32 %f4, 0f00000000;
                add.u64 %rd14, %rd13, %rd5;
                @%p5 ld.global.f32 %f3, [%rd14];
                add.u64 %rd14, %rd13, %rd6;
                @%p5 ld.global.f32 %f4, [%rd14];
                mul.lo.u32 %r34, %r7, {{Stride * 4}};
                add.u32 %r35, %r34, %r16;
                add.u32 %r36, %r34, %r17;
            """);
        for (int kk = 0; kk < Keys; kk++)
        {
            // Key p = %r14 + kk: s = q_lane · k_p, dp = dO_lane · v_p (k, v rows broadcast from global memory); seen
            // when the lane's row range holds p.
            s.AppendLine(CultureInfo.InvariantCulture, $$"""
                    add.u32 %r37, %r14, {{kk}};
                    setp.lt.u32 %p6, %r37, %r3;
                    sub.u32 %r39, %r3, 1;
                    min.u32 %r38, %r37, %r39;
                    mul.lo.u32 %r40, %r38, %r4;
                    mul.wide.u32 %rd15, %r40, 4;
                    add.u64 %rd16, %rd15, %rd2;
                    add.u64 %rd17, %rd15, %rd3;
                    mov.f32 %f5, 0f00000000;
                    mov.f32 %f6, 0f00000000;
                    mov.u32 %r41, %r35;
                    mov.u32 %r42, %r36;
                    mov.u32 %r43, 0;
                SD{{kk}}:
                    setp.ge.u32 %p7, %r43, %r4;
                    @%p7 bra SD{{kk}}_END;
                    ld.shared.f32 %f7, [%r41];
                    ld.global.f32 %f8, [%rd16];
                    fma.rn.f32 %f5, %f7, %f8, %f5;
                    ld.shared.f32 %f9, [%r42];
                    ld.global.f32 %f10, [%rd17];
                    fma.rn.f32 %f6, %f9, %f10, %f6;
                    add.u32 %r41, %r41, 4;
                    add.u32 %r42, %r42, 4;
                    add.u64 %rd16, %rd16, 4;
                    add.u64 %rd17, %rd17, 4;
                    add.u32 %r43, %r43, 1;
                    bra SD{{kk}};
                SD{{kk}}_END:
                    setp.ge.u32 %p8, %r37, %lo;
                    setp.lt.and.u32 %p8, %r37, %hi, %p8;
                    and.pred %p8, %p8, %p5;
                    and.pred %p8, %p8, %p6;
                    mul.f32 %f11, %f5, %f31;
                    {{SoftcapPtx("%f11", "%f30", "%p15", "%f26")}}
                    mov.f32 %f25, %f11;
                    sub.f32 %f11, %f11, %f3;
                    mul.f32 %f11, %f11, 0f3FB8AA3B;
                    ex2.approx.ftz.f32 %f11, %f11;
                    selp.f32 %sp{{kk}}, %f11, 0f00000000, %p8;
                    sub.f32 %f12, %f6, %f4;
                    mul.f32 %sd{{kk}}, %sp{{kk}}, %f12;
                    {{SoftcapSlopePtx(string.Create(CultureInfo.InvariantCulture, $"%sd{kk}"), "%f25", "%f30", "%p15", "%f26")}}
                """);
        }

        // dV_p += Σ_r P[r, p] dO[r], dK_p += Σ_r dS[r, p] Q[r] with the lanes splitting the dimension.
        s.AppendLine(CultureInfo.InvariantCulture, $$"""
                mov.u32 %r44, 0;
            RR:
                setp.ge.u32 %p9, %r44, {{QT}};
                @%p9 bra RR_END;
                mul.lo.u32 %r45, %r44, {{Stride * 4}};
                shl.b32 %r46, %r7, 2;
                add.u32 %r45, %r45, %r46;
                add.u32 %r46, %r45, %r16;
                add.u32 %r47, %r45, %r17;
            """);
        for (int j = 0; j < 4; j++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    @%dv{j} ld.shared.f32 %f{13 + j}, [%r46+{128 * j}];");
            s.AppendLine(CultureInfo.InvariantCulture, $"    @%dv{j} ld.shared.f32 %f{17 + j}, [%r47+{128 * j}];");
        }

        for (int kk = 0; kk < Keys; kk++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    shfl.sync.idx.b32 %f21, %sp{kk}, %r44, 31, 0xffffffff;");
            s.AppendLine(CultureInfo.InvariantCulture, $"    shfl.sync.idx.b32 %f22, %sd{kk}, %r44, 31, 0xffffffff;");
            for (int j = 0; j < 4; j++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    fma.rn.f32 %gv{kk * 4 + j}, %f21, %f{17 + j}, %gv{kk * 4 + j};");
                s.AppendLine(CultureInfo.InvariantCulture, $"    fma.rn.f32 %gk{kk * 4 + j}, %f22, %f{13 + j}, %gk{kk * 4 + j};");
            }
        }

        s.AppendLine(CultureInfo.InvariantCulture, $$"""
                add.u32 %r44, %r44, 1;
                bra RR;
            RR_END:
            QT_NEXT:
                add.u32 %r19, %r19, {{QT}};
                bra QT;
            QT_END:
                add.u32 %h, %h, 1;
                bra HL;
            HL_END:
            """);
        for (int kk = 0; kk < Keys; kk++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $$"""
                    add.u32 %r37, %r14, {{kk}};
                    setp.ge.u32 %p10, %r37, %r3;
                    @%p10 bra KW{{kk}};
                    mad.lo.u32 %r40, %r37, %r4, %r7;
                    mul.wide.u32 %rd18, %r40, 4;
                    add.u64 %rd19, %rd18, %rd8;
                    add.u64 %rd20, %rd18, %rd9;
                """);
            for (int j = 0; j < 4; j++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $$"""
                        @%dv{{j}} ld.global.f32 %f23, [%rd19+{{128 * j}}];
                        @%dv{{j}} fma.rn.f32 %f23, %gk{{kk * 4 + j}}, %f31, %f23;
                        @%dv{{j}} st.global.f32 [%rd19+{{128 * j}}], %f23;
                        @%dv{{j}} ld.global.f32 %f24, [%rd20+{{128 * j}}];
                        @%dv{{j}} add.f32 %f24, %f24, %gv{{kk * 4 + j}};
                        @%dv{{j}} st.global.f32 [%rd20+{{128 * j}}], %f24;
                    """);
            }

            s.AppendLine(CultureInfo.InvariantCulture, $"KW{kk}:");
        }

        s.AppendLine("""
                ret;
            }
            """);
        sb.AppendLine(s.ToString());
    }

    private static void AttentionSpansBackwardQueries(StringBuilder sb)
    {
        const int Rows = 8, KT = 32, Stride = 33, D = FlashMaxDim;
        var s = new StringBuilder();
        s.AppendLine(CultureInfo.InvariantCulture, $$"""
            .visible .entry {{SpanBackwardQueriesName}}(
            {{SpanBackwardParameters}}
            )
            {
                .reg .pred %p<16>;
                .reg .pred %dv<4>;
                .reg .f32 %gq<{{Rows * 4}}>;
                .reg .f32 %f<32>;
                .reg .f32 %pq<4>;
                .reg .b32 %r<64>;
                .reg .b64 %rd<40>;
                .reg .b32 %lo<{{Rows}}>;
                .reg .b32 %hi<{{Rows}}>;
                .reg .b32 %mylo, %myhi, %from, %to, %tb;
                .shared .align 4 .f32 sbq_k[{{D * Stride}}];
                .shared .align 4 .f32 sbq_v[{{D * Stride}}];
            {{SpanBackwardPrologue}}
                mov.u32 %r13, %ctaid.x;
                shl.b32 %r13, %r13, 5;
                mov.u32 %r55, %ctaid.y;
            {{SpanHeadPointers("%r55")}}
                div.u32 %r56, %r55, %r50;
            {{SpanKvPointers("%r56")}}
                mul.lo.u32 %r15, %r4, {{KT}};
                mov.u32 %r16, sbq_k;
                mov.u32 %r17, sbq_v;
                add.u32 %r57, %r13, %r7;
            {{SpanRowRange("%r57", "%mylo", "%myhi", "%p5")}}
            """);

        // Row warp·8 + i's range from the lane that read it; the block's keys: the smallest start and largest end of its
        // rows' non-empty ranges (an empty range counts as [keyRows, 0)); every warp reads the same 32 rows.
        for (int i = 0; i < Rows; i++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    mad.lo.u32 %r58, %r8, {Rows}, {i};");
            s.AppendLine(CultureInfo.InvariantCulture, $"    shfl.sync.idx.b32 %lo{i}, %mylo, %r58, 31, 0xffffffff;");
            s.AppendLine(CultureInfo.InvariantCulture, $"    shfl.sync.idx.b32 %hi{i}, %myhi, %r58, 31, 0xffffffff;");
        }

        s.AppendLine("""
                setp.gt.u32 %p3, %myhi, %mylo;
                selp.b32 %from, %mylo, %r3, %p3;
                selp.b32 %to, %myhi, 0, %p3;
            """);
        foreach (int offset in new[] { 16, 8, 4, 2, 1 })
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    shfl.sync.bfly.b32 %r59, %from, {offset}, 31, 0xffffffff;");
            s.AppendLine("    min.u32 %from, %from, %r59;");
            s.AppendLine(CultureInfo.InvariantCulture, $"    shfl.sync.bfly.b32 %r59, %to, {offset}, 31, 0xffffffff;");
            s.AppendLine("    max.u32 %to, %to, %r59;");
        }

        for (int j = 0; j < 4; j++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    add.u32 %r18, %r7, {32 * j};");
            s.AppendLine(CultureInfo.InvariantCulture, $"    setp.lt.u32 %dv{j}, %r18, %r4;");
        }

        for (int i = 0; i < Rows * 4; i++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    mov.f32 %gq{i}, 0f00000000;");
        }

        s.AppendLine(CultureInfo.InvariantCulture, $$"""
                mov.u32 %r19, %from;
            KT:
                setp.ge.u32 %p1, %r19, %to;
                @%p1 bra KT_END;
                bar.sync 0;
                mov.u32 %r25, %r6;
            KL:
                setp.ge.u32 %p3, %r25, %r15;
                @%p3 bra KL_END;
                div.u32 %r26, %r25, %r4;
                rem.u32 %r27, %r25, %r4;
                add.u32 %r28, %r19, %r26;
                setp.lt.u32 %p4, %r28, %r3;
                mad.lo.u32 %r29, %r28, %r4, %r27;
                mul.wide.u32 %rd10, %r29, 4;
                add.u64 %rd11, %rd10, %rd2;
                add.u64 %rd12, %rd10, %rd3;
                mov.f32 %f1, 0f00000000;
                mov.f32 %f2, 0f00000000;
                @%p4 ld.global.f32 %f1, [%rd11];
                @%p4 ld.global.f32 %f2, [%rd12];
                mad.lo.u32 %r30, %r27, {{Stride}}, %r26;
                shl.b32 %r30, %r30, 2;
                add.u32 %r31, %r30, %r16;
                st.shared.f32 [%r31], %f1;
                add.u32 %r31, %r30, %r17;
                st.shared.f32 [%r31], %f2;
                add.u32 %r25, %r25, 128;
                bra KL;
            KL_END:
                bar.sync 0;
                add.u32 %r32, %r19, %r7;
                setp.lt.u32 %p5, %r32, %r3;
                shl.b32 %r34, %r7, 2;
            """);
        for (int i = 0; i < Rows; i++)
        {
            // Row r = block row + warp·8 + i: s = q_r · k_lane, dp = dO_r · v_lane (q, dO rows broadcast from global);
            // seen when the row's range holds key %r32.
            s.AppendLine(CultureInfo.InvariantCulture, $$"""
                    mad.lo.u32 %r35, %r8, {{Rows}}, {{i}};
                    add.u32 %r35, %r35, %r13;
                    setp.lt.u32 %p6, %r35, %r1;
                    sub.u32 %r36, %r1, 1;
                    min.u32 %r36, %r35, %r36;
                    mul.lo.u32 %r38, %r36, %r4;
                    mul.wide.u32 %rd13, %r38, 4;
                    add.u64 %rd14, %rd13, %rd1;
                    add.u64 %rd15, %rd13, %rd4;
                    mul.wide.u32 %rd16, %r36, 4;
                    add.u64 %rd17, %rd16, %rd5;
                    ld.global.f32 %f3, [%rd17];
                    add.u64 %rd17, %rd16, %rd6;
                    ld.global.f32 %f4, [%rd17];
                    mov.f32 %f5, 0f00000000;
                    mov.f32 %f6, 0f00000000;
                    add.u32 %r41, %r34, %r16;
                    add.u32 %r42, %r34, %r17;
                    mov.u32 %r43, 0;
                SQ{{i}}:
                    setp.ge.u32 %p7, %r43, %r4;
                    @%p7 bra SQ{{i}}_END;
                    ld.global.f32 %f7, [%rd14];
                    ld.shared.f32 %f8, [%r41];
                    fma.rn.f32 %f5, %f7, %f8, %f5;
                    ld.global.f32 %f9, [%rd15];
                    ld.shared.f32 %f10, [%r42];
                    fma.rn.f32 %f6, %f9, %f10, %f6;
                    add.u64 %rd14, %rd14, 4;
                    add.u64 %rd15, %rd15, 4;
                    add.u32 %r41, %r41, {{Stride * 4}};
                    add.u32 %r42, %r42, {{Stride * 4}};
                    add.u32 %r43, %r43, 1;
                    bra SQ{{i}};
                SQ{{i}}_END:
                    setp.ge.u32 %p8, %r32, %lo{{i}};
                    setp.lt.and.u32 %p8, %r32, %hi{{i}}, %p8;
                    and.pred %p8, %p8, %p5;
                    and.pred %p8, %p8, %p6;
                    mul.f32 %f11, %f5, %f31;
                    {{SoftcapPtx("%f11", "%f30", "%p15", "%f26")}}
                    mov.f32 %f25, %f11;
                    sub.f32 %f11, %f11, %f3;
                    mul.f32 %f11, %f11, 0f3FB8AA3B;
                    ex2.approx.ftz.f32 %f11, %f11;
                    selp.f32 %f11, %f11, 0f00000000, %p8;
                    sub.f32 %f12, %f6, %f4;
                    mul.f32 %f12, %f11, %f12;
                    {{SoftcapSlopePtx("%f12", "%f25", "%f30", "%p15", "%f26")}}
                    mov.u32 %r44, 0;
                    mul.lo.u32 %r46, %r7, {{Stride * 4}};
                    add.u32 %r46, %r46, %r16;
                PQ{{i}}:
                    setp.ge.u32 %p9, %r44, {{KT}};
                    @%p9 bra PQ{{i}}_END;
                    shfl.sync.idx.b32 %f13, %f12, %r44, 31, 0xffffffff;
                """);
            // A register per dimension slot, all four loaded before any is used (see attention_flash_f32's PV loop).
            for (int j = 0; j < 4; j++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    @%dv{j} ld.shared.f32 %pq{j}, [%r46+{32 * Stride * 4 * j}];");
            }

            for (int j = 0; j < 4; j++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    @%dv{j} fma.rn.f32 %gq{i * 4 + j}, %f13, %pq{j}, %gq{i * 4 + j};");
            }

            s.AppendLine(CultureInfo.InvariantCulture, $$"""
                    add.u32 %r46, %r46, 4;
                    add.u32 %r44, %r44, 1;
                    bra PQ{{i}};
                PQ{{i}}_END:
                """);
        }

        s.AppendLine(CultureInfo.InvariantCulture, $$"""
                add.u32 %r19, %r19, {{KT}};
                bra KT;
            KT_END:
            """);
        for (int i = 0; i < Rows; i++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $$"""
                    mad.lo.u32 %r35, %r8, {{Rows}}, {{i}};
                    add.u32 %r35, %r35, %r13;
                    setp.ge.u32 %p10, %r35, %r1;
                    @%p10 bra QW{{i}};
                    mad.lo.u32 %r38, %r35, %r4, %r7;
                    mul.wide.u32 %rd18, %r38, 4;
                    add.u64 %rd19, %rd18, %rd7;
                """);
            for (int j = 0; j < 4; j++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $$"""
                        @%dv{{j}} ld.global.f32 %f23, [%rd19+{{128 * j}}];
                        @%dv{{j}} fma.rn.f32 %f23, %gq{{i * 4 + j}}, %f31, %f23;
                        @%dv{{j}} st.global.f32 [%rd19+{{128 * j}}], %f23;
                    """);
            }

            s.AppendLine(CultureInfo.InvariantCulture, $"QW{i}:");
        }

        s.AppendLine("""
                ret;
            }
            """);
        sb.AppendLine(s.ToString());
    }
}
