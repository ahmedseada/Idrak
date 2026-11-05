// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;

namespace Idrak.Gpu.Cuda;

// PTX for the mixture-of-experts operations (plan 37, R0c; Backend.Routing.cs). Their order and arithmetic are those of
// Vulkan's (VulkanKernels.Routing.cs), HIP's (GcnKernels.Routing.cs) and the CPU's (CpuBackend.Routing.cs), whose routing
// and products they equal bit for bit. Indices, offsets, choices and counts are int32 (T1b). One thread a unit (Elementwise):
// - route_choose_f32: a thread per token. Its k most probable experts are found as the CPU walks a row: each expert is
//   inserted after every chosen one at least as probable, so ties go to the lowest index. The list is kept in the
//   token's own slots of `choices` (no limit on k), then sorted into increasing expert order. The weights' sum is added
//   in that order, and each weight is divided by it (div.rn: correctly rounded) when asked.
// - route_counts_f32: a thread per expert.
// - route_offsets_f32: one thread; the counts are added in expert order.
// - route_place_f32: a thread per expert; its rows are placed in slot order.
// - expert_gated_f32 and expert_matmul_f32: one launch per expert, a thread per output of every row. A thread past its
//   expert's rows × n leaves at once; the count is read from the offsets on the device. Each output is one fused
//   multiply-add chain over k from zero, in order (rule 86, the CPU's product). The bias is added after it with add.rn,
//   never contracted; a null bias pointer means none. Then comes the gated activation as gated_act_f32 computes it
//   (GatedActivation).
//
// | Kernel             | Arguments                                                              | Thread         |
// |--------------------|------------------------------------------------------------------------|----------------|
// | route_choose_f32   | probabilities, weights, choices; experts, k, normalize; n (tokens)      | a token        |
// | route_counts_f32   | choices, counts; slots; n (experts)                                    | an expert      |
// | route_offsets_f32  | counts, offsets; slots, experts; n (1)                                 | all            |
// | route_place_f32    | choices, offsets, tokenOfRow, slotRow; slots, k; n (experts)           | an expert      |
// | expert_gated_f32   | x, offsets, gate, gatebias, up, upbias, hidden; expert, k, cols, kind; n | an output    |
// | expert_matmul_f32  | x, offsets, weight, bias, y; expert, k, cols; n (rows · cols)          | an output      |
internal static partial class PtxKernels
{
    /// <summary>The kernels of this file (in the main module).</summary>
    public static readonly string[] RoutingNames = ["route_choose_f32", "route_counts_f32", "route_offsets_f32", "route_place_f32", "expert_gated_f32", "expert_matmul_f32"];

    // The first row of expert %s_expert (%r4), and its outputs (rows · cols) in %r6; a thread past them leaves. Then the
    // thread's row (%r7) and column (%r8), x's row in %rd2 and the column's byte offset in %rd3, cols · 4 in %rd6.
    private const string ExpertOutput = """
        mul.wide.u32 %rd1, %s_expert, 4;
        add.u64 %rd1, %b_offsets, %rd1;
        ld.global.u32 %r4, [%rd1];
        ld.global.u32 %r5, [%rd1+4];
        sub.u32 %r6, %r5, %r4;
        mul.lo.u32 %r6, %r6, %s_cols;
        setp.ge.u32 %p1, %i, %r6;
        @%p1 bra DONE;
        div.u32 %r7, %i, %s_cols;
        rem.u32 %r8, %i, %s_cols;
        add.u32 %r7, %r7, %r4;
        mul.lo.u32 %r9, %r7, %s_k;
        mul.wide.u32 %rd2, %r9, 4;
        add.u64 %rd2, %b_x, %rd2;
        mul.wide.u32 %rd3, %r8, 4;
        mul.wide.u32 %rd6, %s_cols, 4;
        """;

    // The output's byte offset (row · cols + col) in %rd8.
    private const string ExpertStore = """
        mul.lo.u32 %r11, %r7, %s_cols;
        add.u32 %r11, %r11, %r8;
        mul.wide.u32 %rd8, %r11, 4;
        """;

    private static void BuildRouting(StringBuilder sb)
    {
        // probabilities[row + expert] of the expert in %r11 into %f2 (row = %r4).
        const string Probability = """
            add.u32 %r12, %r4, %r11;
            mul.wide.u32 %rd3, %r12, 4;
            add.u64 %rd3, %b_probabilities, %rd3;
            ld.global.f32 %f2, [%rd3];
            """;

        // choices[first + %r10] into %r11 (first = %r5), its address in %rd2.
        const string Choice = """
            add.u32 %r10, %r5, %r10;
            mul.wide.u32 %rd2, %r10, 4;
            add.u64 %rd2, %b_choices, %rd2;
            ld.global.u32 %r11, [%rd2];
            """;

        Elementwise(sb, "route_choose_f32", ["probabilities", "weights", "choices"], [("u32", "experts"), ("u32", "k"), ("u32", "normalize")],
            $$"""
            mul.lo.u32 %r4, %i, %s_experts;
            mul.lo.u32 %r5, %i, %s_k;
            mov.u32 %r6, 0;
            mov.u32 %r7, 0;
            CH_EXPERT:
            setp.ge.u32 %p1, %r7, %s_experts;
            @%p1 bra CH_SORT;
            add.u32 %r8, %r4, %r7;
            mul.wide.u32 %rd1, %r8, 4;
            add.u64 %rd1, %b_probabilities, %rd1;
            ld.global.f32 %f1, [%rd1];
            mov.u32 %r9, %r6;
            CH_WALK:
            setp.eq.u32 %p2, %r9, 0;
            @%p2 bra CH_PLACED;
            sub.u32 %r10, %r9, 1;
            {{Choice}}
            {{Probability}}
            setp.gt.f32 %p3, %f1, %f2;
            @!%p3 bra CH_PLACED;
            sub.u32 %r9, %r9, 1;
            bra CH_WALK;
            CH_PLACED:
            setp.ge.u32 %p4, %r9, %s_k;
            @%p4 bra CH_NEXT;
            sub.u32 %r13, %s_k, 1;
            min.u32 %r13, %r6, %r13;
            CH_SHIFT:
            setp.le.u32 %p5, %r13, %r9;
            @%p5 bra CH_PUT;
            add.u32 %r10, %r5, %r13;
            mul.wide.u32 %rd2, %r10, 4;
            add.u64 %rd2, %b_choices, %rd2;
            add.s64 %rd4, %rd2, -4;
            ld.global.u32 %r11, [%rd4];
            st.global.u32 [%rd2], %r11;
            sub.u32 %r13, %r13, 1;
            bra CH_SHIFT;
            CH_PUT:
            add.u32 %r10, %r5, %r9;
            mul.wide.u32 %rd2, %r10, 4;
            add.u64 %rd2, %b_choices, %rd2;
            st.global.u32 [%rd2], %r7;
            add.u32 %r6, %r6, 1;
            min.u32 %r6, %r6, %s_k;
            CH_NEXT:
            add.u32 %r7, %r7, 1;
            bra CH_EXPERT;
            CH_SORT:
            mov.u32 %r14, 1;
            CH_SORT_NEXT:
            setp.ge.u32 %p1, %r14, %s_k;
            @%p1 bra CH_SUM;
            mov.u32 %r10, %r14;
            {{Choice}}
            mov.u32 %r15, %r11;
            mov.u32 %r16, %r14;
            CH_SORT_WALK:
            setp.eq.u32 %p2, %r16, 0;
            @%p2 bra CH_SORT_PUT;
            add.s64 %rd4, %rd2, -4;
            ld.global.u32 %r11, [%rd4];
            setp.le.s32 %p3, %r11, %r15;
            @%p3 bra CH_SORT_PUT;
            st.global.u32 [%rd2], %r11;
            mov.u64 %rd2, %rd4;
            sub.u32 %r16, %r16, 1;
            bra CH_SORT_WALK;
            CH_SORT_PUT:
            st.global.u32 [%rd2], %r15;
            add.u32 %r14, %r14, 1;
            bra CH_SORT_NEXT;
            CH_SUM:
            mov.f32 %f3, {{Zero}};
            mov.u32 %r17, 0;
            CH_SUM_NEXT:
            setp.ge.u32 %p1, %r17, %s_k;
            @%p1 bra CH_WEIGHTS;
            mov.u32 %r10, %r17;
            {{Choice}}
            {{Probability}}
            add.rn.f32 %f3, %f3, %f2;
            add.u32 %r17, %r17, 1;
            bra CH_SUM_NEXT;
            CH_WEIGHTS:
            setp.ne.u32 %p6, %s_normalize, 0;
            mov.u32 %r17, 0;
            CH_WEIGHT_NEXT:
            setp.ge.u32 %p1, %r17, %s_k;
            @%p1 bra DONE;
            mov.u32 %r10, %r17;
            {{Choice}}
            {{Probability}}
            div.rn.f32 %f4, %f2, %f3;
            selp.f32 %f4, %f4, %f2, %p6;
            add.u32 %r10, %r5, %r17;
            mul.wide.u32 %rd5, %r10, 4;
            add.u64 %rd5, %b_weights, %rd5;
            st.global.f32 [%rd5], %f4;
            add.u32 %r17, %r17, 1;
            bra CH_WEIGHT_NEXT;
            """);

        // counts[e] = the slots whose choice is e (every thread reads the same slot at once: one load a warp).
        Elementwise(sb, "route_counts_f32", ["choices", "counts"], [("u32", "slots")],
            """
            mov.u32 %r4, 0;
            mov.u32 %r5, 0;
            mov.u64 %rd1, %b_choices;
            RC_SLOT:
            setp.ge.u32 %p1, %r5, %s_slots;
            @%p1 bra RC_DONE;
            ld.global.u32 %r6, [%rd1];
            setp.eq.u32 %p2, %r6, %i;
            selp.u32 %r7, 1, 0, %p2;
            add.u32 %r4, %r4, %r7;
            add.u64 %rd1, %rd1, 4;
            add.u32 %r5, %r5, 1;
            bra RC_SLOT;
            RC_DONE:
            st.global.u32 [%a_counts], %r4;
            """);

        // offsets[e] = the counts before e, added in expert order; offsets[experts] = slots.
        Elementwise(sb, "route_offsets_f32", ["counts", "offsets"], [("u32", "slots"), ("u32", "experts")],
            """
            mov.u32 %r4, 0;
            mov.u32 %r5, 0;
            mov.u64 %rd1, %b_counts;
            mov.u64 %rd2, %b_offsets;
            RO_EXPERT:
            setp.ge.u32 %p1, %r5, %s_experts;
            @%p1 bra RO_END;
            st.global.u32 [%rd2], %r4;
            ld.global.u32 %r6, [%rd1];
            add.u32 %r4, %r4, %r6;
            add.u64 %rd1, %rd1, 4;
            add.u64 %rd2, %rd2, 4;
            add.u32 %r5, %r5, 1;
            bra RO_EXPERT;
            RO_END:
            st.global.u32 [%rd2], %s_slots;
            """);

        // Expert e's rows in slot order: tokenOfRow[row] = slot / k and slotRow[slot] = row, row from offsets[e] on.
        Elementwise(sb, "route_place_f32", ["choices", "offsets", "tokenOfRow", "slotRow"], [("u32", "slots"), ("u32", "k")],
            """
            ld.global.u32 %r4, [%a_offsets];
            mov.u32 %r5, 0;
            mov.u32 %r8, 0;
            mov.u32 %r9, 0;
            mov.u64 %rd1, %b_choices;
            RP_SLOT:
            setp.ge.u32 %p1, %r5, %s_slots;
            @%p1 bra DONE;
            ld.global.u32 %r6, [%rd1];
            setp.ne.u32 %p2, %r6, %i;
            @%p2 bra RP_NEXT;
            mul.wide.u32 %rd2, %r4, 4;
            add.u64 %rd2, %b_tokenOfRow, %rd2;
            st.global.u32 [%rd2], %r8;
            mul.wide.u32 %rd3, %r5, 4;
            add.u64 %rd3, %b_slotRow, %rd3;
            st.global.u32 [%rd3], %r4;
            add.u32 %r4, %r4, 1;
            RP_NEXT:
            add.u32 %r9, %r9, 1;
            setp.eq.u32 %p3, %r9, %s_k;
            @%p3 mov.u32 %r9, 0;
            @%p3 add.u32 %r8, %r8, 1;
            add.u64 %rd1, %rd1, 4;
            add.u32 %r5, %r5, 1;
            bra RP_SLOT;
            """);

        // hidden[row, col] = act(x[row]·gate[:, col] + gatebias[col]) · (x[row]·up[:, col] + upbias[col]) over expert
        // %s_expert's rows, act by %s_kind as gated_act_f32's.
        Elementwise(sb, "expert_gated_f32", ["x", "offsets", "gate", "gatebias", "up", "upbias", "hidden"],
            [("u32", "expert"), ("u32", "k"), ("u32", "cols"), ("u32", "kind")],
            ExpertOutput + "\n" + $$"""
            add.u64 %rd4, %b_gate, %rd3;
            add.u64 %rd5, %b_up, %rd3;
            mov.f32 %f20, {{Zero}};
            mov.f32 %f21, {{Zero}};
            mov.u32 %r10, 0;
            EG_K:
            setp.ge.u32 %p2, %r10, %s_k;
            @%p2 bra EG_GATE_BIAS;
            ld.global.f32 %f22, [%rd2];
            ld.global.f32 %f23, [%rd4];
            ld.global.f32 %f24, [%rd5];
            fma.rn.f32 %f20, %f22, %f23, %f20;
            fma.rn.f32 %f21, %f22, %f24, %f21;
            add.u64 %rd2, %rd2, 4;
            add.u64 %rd4, %rd4, %rd6;
            add.u64 %rd5, %rd5, %rd6;
            add.u32 %r10, %r10, 1;
            bra EG_K;
            EG_GATE_BIAS:
            ld.param.u64 %rd7, [p_gatebias];
            setp.eq.u64 %p3, %rd7, 0;
            @%p3 bra EG_UP_BIAS;
            add.u64 %rd7, %b_gatebias, %rd3;
            ld.global.f32 %f22, [%rd7];
            add.rn.f32 %f20, %f20, %f22;
            EG_UP_BIAS:
            ld.param.u64 %rd7, [p_upbias];
            setp.eq.u64 %p3, %rd7, 0;
            @%p3 bra EG_ACT;
            add.u64 %rd7, %b_upbias, %rd3;
            ld.global.f32 %f22, [%rd7];
            add.rn.f32 %f21, %f21, %f22;
            EG_ACT:
            mov.f32 %f1, %f20;
            mov.f32 %f2, %f21;
            """ + "\n" + GatedActivation + "\n" + ExpertStore + "\n" + """
            mul.f32 %f5, %f5, %f19;
            add.u64 %rd8, %b_hidden, %rd8;
            st.global.f32 [%rd8], %f5;
            """);

        // y[row, col] = x[row]·weight[:, col] + bias[col] over expert %s_expert's rows.
        Elementwise(sb, "expert_matmul_f32", ["x", "offsets", "weight", "bias", "y"], [("u32", "expert"), ("u32", "k"), ("u32", "cols")],
            ExpertOutput + "\n" + $$"""
            add.u64 %rd4, %b_weight, %rd3;
            mov.f32 %f20, {{Zero}};
            mov.u32 %r10, 0;
            EM_K:
            setp.ge.u32 %p2, %r10, %s_k;
            @%p2 bra EM_BIAS;
            ld.global.f32 %f22, [%rd2];
            ld.global.f32 %f23, [%rd4];
            fma.rn.f32 %f20, %f22, %f23, %f20;
            add.u64 %rd2, %rd2, 4;
            add.u64 %rd4, %rd4, %rd6;
            add.u32 %r10, %r10, 1;
            bra EM_K;
            EM_BIAS:
            ld.param.u64 %rd7, [p_bias];
            setp.eq.u64 %p3, %rd7, 0;
            @%p3 bra EM_STORE;
            add.u64 %rd7, %b_bias, %rd3;
            ld.global.f32 %f22, [%rd7];
            add.rn.f32 %f20, %f20, %f22;
            EM_STORE:
            """ + "\n" + ExpertStore + "\n" + """
            add.u64 %rd8, %b_y, %rd8;
            st.global.f32 [%rd8], %f20;
            """);
    }
}
