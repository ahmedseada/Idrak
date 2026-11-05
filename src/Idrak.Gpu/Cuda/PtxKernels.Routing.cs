// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
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
// - expert_gated_f32 and expert_matmul_f32: one launch for up to ExpertsPerLaunch experts (plan 37, R0d). The driver
//   API has no launch whose size the device writes: dynamic parallelism needs the device runtime linked, and graph
//   conditional nodes need a newer driver. So the launch walks its experts one after another. Each expert's addresses
//   come from its slot of the parameters, its rows from the offsets on the device, and the whole grid runs over its
//   rows × n outputs grid-stride. An expert without rows costs two reads, not a launch.
//   Each output is one fused multiply-add chain over k from zero, in order (rule 86, the CPU's product). The bias is
//   added after it with add.rn, never contracted; a null bias pointer means none. Then comes the gated activation as
//   gated_act_f32 computes it (GatedActivation).
//
// | Kernel             | Arguments                                                              | Thread         |
// |--------------------|------------------------------------------------------------------------|----------------|
// | route_choose_f32   | probabilities, weights, choices; experts, k, normalize; n (tokens)      | a token        |
// | route_counts_f32   | choices, counts; slots; n (experts)                                    | an expert      |
// | route_offsets_f32  | counts, offsets; slots, experts; n (1)                                 | all            |
// | route_place_f32    | choices, offsets, tokenOfRow, slotRow; slots, k; n (experts)           | an expert      |
// | expert_gated_f32   | x, offsets, hidden; first, experts, k, cols, kind; per slot gate, up, gatebias, upbias | grid-stride |
// | expert_matmul_f32  | x, offsets, y; first, experts, k, cols; per slot weight, bias          | grid-stride    |
internal static partial class PtxKernels
{
    /// <summary>The kernels of this file (in the main module).</summary>
    public static readonly string[] RoutingNames = ["route_choose_f32", "route_counts_f32", "route_offsets_f32", "route_place_f32", "expert_gated_f32", "expert_matmul_f32"];

    /// <summary>
    /// The experts one expert launch takes: each expert's addresses are parameters of their own (a slot), read as the
    /// launch reaches the expert. A mixture with more experts takes a launch for each such group.
    /// </summary>
    public const int ExpertsPerLaunch = 16;

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

        // hidden[row, col] = act(x[row]·gate[:, col] + gatebias[col]) · (x[row]·up[:, col] + upbias[col]) over each expert's
        // rows, act by %s_kind as gated_act_f32's. The expert's gate in %rd14, up in %rd15, the biases as passed in %rd12, %rd13.
        ExpertWalk(sb, "expert_gated_f32", ["gate", "up", "gatebias", "upbias"], gated: true, $$"""
            add.u64 %rd4, %rd14, %rd3;
            add.u64 %rd5, %rd15, %rd3;
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
            setp.eq.u64 %p3, %rd12, 0;
            @%p3 bra EG_UP_BIAS;
            cvta.to.global.u64 %rd7, %rd12;
            add.u64 %rd7, %rd7, %rd3;
            ld.global.f32 %f22, [%rd7];
            add.rn.f32 %f20, %f20, %f22;
            EG_UP_BIAS:
            setp.eq.u64 %p3, %rd13, 0;
            @%p3 bra EG_ACT;
            cvta.to.global.u64 %rd7, %rd13;
            add.u64 %rd7, %rd7, %rd3;
            ld.global.f32 %f22, [%rd7];
            add.rn.f32 %f21, %f21, %f22;
            EG_ACT:
            mov.f32 %f1, %f20;
            mov.f32 %f2, %f21;
            """ + "\n" + GatedActivation + "\n" + ExpertStore + "\n" + """
            mul.f32 %f5, %f5, %f19;
            add.u64 %rd8, %b_out, %rd8;
            st.global.f32 [%rd8], %f5;
            """);

        // y[row, col] = x[row]·weight[:, col] + bias[col] over each expert's rows: the weight in %rd14, the bias as passed in %rd11.
        ExpertWalk(sb, "expert_matmul_f32", ["weight", "bias"], gated: false, $$"""
            add.u64 %rd4, %rd14, %rd3;
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
            setp.eq.u64 %p3, %rd11, 0;
            @%p3 bra EM_STORE;
            cvta.to.global.u64 %rd7, %rd11;
            add.u64 %rd7, %rd7, %rd3;
            ld.global.f32 %f22, [%rd7];
            add.rn.f32 %f20, %f20, %f22;
            EM_STORE:
            """ + "\n" + ExpertStore + "\n" + """
            add.u64 %rd8, %b_out, %rd8;
            st.global.f32 [%rd8], %f20;
            """);
    }

    // An expert launch: parameters x, offsets, the output (p_out); first, experts, k, cols (and kind); then ExpertsPerLaunch
    // slots of `pointers` (p_{name}{slot}). The experts first … first + experts - 1 run one after another:
    // - slot j's addresses go into %rd10 to %rd13, in the order of `pointers`. The first two are also converted to global
    //   addresses in %rd14 and %rd15 (the second is used only where it is not a bias).
    // - the expert's first row goes into %r4, and its outputs (rows · cols) into %r6.
    // - then `body` runs for each of its outputs, grid-stride: the output's row in %r7, its column in %r8, x's row in
    //   %rd2, the column's byte offset in %rd3, and cols · 4 in %rd6.
    private static void ExpertWalk(StringBuilder sb, string name, string[] pointers, bool gated, string body)
    {
        sb.Append(".visible .entry ").Append(name).Append("(\n");
        sb.Append("    .param .u64 p_x,\n    .param .u64 p_offsets,\n    .param .u64 p_out,\n");
        sb.Append("    .param .u32 p_first,\n    .param .u32 p_experts,\n    .param .u32 p_k,\n    .param .u32 p_cols");
        if (gated)
        {
            sb.Append(",\n    .param .u32 p_kind");
        }

        for (int j = 0; j < ExpertsPerLaunch; j++)
        {
            foreach (string pointer in pointers)
            {
                sb.Append(CultureInfo.InvariantCulture, $",\n    .param .u64 p_{pointer}{j}");
            }
        }

        sb.AppendLine().AppendLine(")").AppendLine("{");
        sb.AppendLine("    .reg .pred %p<16>;");
        sb.AppendLine("    .reg .f32 %f<32>;");
        sb.AppendLine("    .reg .b32 %r<32>;");
        sb.AppendLine("    .reg .b64 %rd<24>;");
        sb.AppendLine("    .reg .u32 %s_first, %s_experts, %s_k, %s_cols" + (gated ? ", %s_kind;" : ";"));
        sb.AppendLine("    .reg .u64 %b_x, %b_offsets, %b_out;");
        var text = new StringBuilder($$"""
            mov.u32 %r1, %ctaid.x;
            mov.u32 %r2, %ntid.x;
            mov.u32 %r3, %tid.x;
            mad.lo.u32 %r28, %r1, %r2, %r3;
            mov.u32 %r1, %nctaid.x;
            mul.lo.u32 %r29, %r1, %r2;
            ld.param.u64 %b_x, [p_x];
            cvta.to.global.u64 %b_x, %b_x;
            ld.param.u64 %b_offsets, [p_offsets];
            cvta.to.global.u64 %b_offsets, %b_offsets;
            ld.param.u64 %b_out, [p_out];
            cvta.to.global.u64 %b_out, %b_out;
            ld.param.u32 %s_first, [p_first];
            ld.param.u32 %s_experts, [p_experts];
            ld.param.u32 %s_k, [p_k];
            ld.param.u32 %s_cols, [p_cols];
            {{(gated ? "ld.param.u32 %s_kind, [p_kind];" : "")}}
            mov.u32 %r27, 0;
            EXPERT:
            setp.ge.u32 %p0, %r27, %s_experts;
            @%p0 bra DONE;
            """);
        text.Append('\n');
        for (int j = 0; j < ExpertsPerLaunch; j++)
        {
            text.Append(CultureInfo.InvariantCulture, $"setp.eq.u32 %p1, %r27, {j};\n");
            for (int p = 0; p < pointers.Length; p++)
            {
                text.Append(CultureInfo.InvariantCulture, $"@%p1 ld.param.u64 %rd{10 + p}, [p_{pointers[p]}{j}];\n");
            }
        }

        text.Append("""
            cvta.to.global.u64 %rd14, %rd10;
            cvta.to.global.u64 %rd15, %rd11;
            add.u32 %r26, %s_first, %r27;
            mul.wide.u32 %rd1, %r26, 4;
            add.u64 %rd1, %b_offsets, %rd1;
            ld.global.u32 %r4, [%rd1];
            ld.global.u32 %r5, [%rd1+4];
            sub.u32 %r6, %r5, %r4;
            mul.lo.u32 %r6, %r6, %s_cols;
            mul.wide.u32 %rd6, %s_cols, 4;
            mov.u32 %r25, %r28;
            OUTPUT:
            setp.ge.u32 %p0, %r25, %r6;
            @%p0 bra NEXT;
            div.u32 %r7, %r25, %s_cols;
            rem.u32 %r8, %r25, %s_cols;
            add.u32 %r7, %r7, %r4;
            mul.lo.u32 %r9, %r7, %s_k;
            mul.wide.u32 %rd2, %r9, 4;
            add.u64 %rd2, %b_x, %rd2;
            mul.wide.u32 %rd3, %r8, 4;
            """).Append('\n').Append(body).Append('\n').Append("""
            add.u32 %r25, %r25, %r29;
            bra OUTPUT;
            NEXT:
            add.u32 %r27, %r27, 1;
            bra EXPERT;
            """);
        AppendBody(sb, text.ToString());
        sb.AppendLine("DONE:");
        sb.AppendLine("    ret;");
        sb.AppendLine("}");
        sb.AppendLine();
    }
}
