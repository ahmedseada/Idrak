// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;

namespace Idrak.Gpu.Cuda;

// PTX for connectionist temporal classification (Backend.CtcLoss and CtcLossBackward): a block per sequence runs α (and
// for the gradient β) over the S = 2L + 1 extended states in log space, in float, the block's threads over the states (a
// fixed stride) with a barrier between steps. The rows (stride `states`: the forward's 2 rows of α; the gradient's 2 rows
// of β, a row of terms and the label groups below) live in dynamic shared memory sized at launch when they fit the
// block's shared memory, else ("_global") in a scratch buffer; the gradient keeps α of every step in global memory
// ([batch, steps, states], then each step's row maximum, [batch, steps]). Per-class sums of the gradient go in a fixed
// order (the blank's through the block's tree, each label's over its occurrences in label order), so the bits do not
// change run to run. The gradient groups the labels once per call (each label's class, and its next occurrence of the
// same class with a flag on all but the first), so a step sums each class by walking its chain, not by scanning the
// labels for every label. Labels are clamped to the classes (the CPU checks them). Parameters: logprobs, targets, meta
// (per sequence: input length, target length, target offset as ints), the outputs and scratch; steps, batch, classes,
// blank, batchfirst, zeroinf, states (the rows' stride: the longest sequence's 2L + 1).
//
// The rows stay near zero so float keeps its precision over long sequences (over hundreds of steps α itself falls to
// about -1,500, where a float holds ~1e-4 and the rounding piles up). Row t is kept as a_t = α_t - P_t: each step
// subtracts the previous row's maximum m_{t-1} (one block reduction, folded into the step's barrier) and the offset
// P_t = m_0 + … + m_{t-1} is summed in double. β likewise: b_t = β_t - R_t, R_t = n_{t+1} + … + n_{T-1} (n: b's row
// maximum). An all -∞ row counts as maximum 0, so it changes no offset. The loss is -(log(e^a_{T-1}(S-1) + e^a_{T-1}(S-2))
// + P_{T-1}) in double, rounded once. The gradient's per-state shares e^(a + b - top) do not see the offsets; its scale
// e^(top_α·β + nll - logprob) takes top_α·β + nll = top_{a+b} + R_t - (P_{T-1} - P_t) - lse_end, with P_{T-1} - P_t =
// m_t + … + m_{T-2} summed in double from the stored maxima (floats, exactly the values the forward subtracted).
internal static partial class PtxKernels
{
    /// <summary>The kernels of this file (in the main module).</summary>
    public static readonly string[] CtcNames = ["ctc_loss_f32", "ctc_loss_global_f32", "ctc_loss_bwd_f32", "ctc_loss_bwd_global_f32"];

    private static readonly string[] CtcScalars = ["steps", "batch", "classes", "blank", "batchfirst", "zeroinf", "states"];

    /// <summary>The CTC kernels' static shared memory (bytes): the reduction tree and the two maxima's warp slots.</summary>
    public static int CtcStaticShared => 4 * (BlockSize + 2 * 2 * (BlockSize / 32));

    /// <summary>Rows (floats, stride states) a sequence keeps: the forward's 2, the gradient's 3 and the label groups (≤ states).</summary>
    public static int CtcRows(bool backward) => backward ? 4 : 2;

    private static void BuildCtc(StringBuilder sb)
    {
        // The shared-memory kernels' rows (CtcRows · states floats, sized at launch): dynamic shared memory is declared
        // once, at module scope (ptxas takes no .extern inside a kernel), and both kernels address it.
        sb.AppendLine(".extern .shared .align 4 .f32 ctc_rows[];");
        foreach (bool shared in new[] { true, false })
        {
            CtcLossKernel(sb, shared);
            CtcBackwardKernel(sb, shared);
        }
    }

    // The common start of a CTC kernel: parameters, registers, the sequence's lengths and offset, its states.
    private static StringBuilder CtcHeader(string name, string[] pointers)
    {
        int threads = BlockSize;
        var p = new StringBuilder();
        p.AppendLine($".visible .entry {name}(");
        foreach (string q in pointers)
        {
            p.Append("    .param .u64 p_").Append(q).Append(",\n");
        }

        for (int i = 0; i < CtcScalars.Length; i++)
        {
            p.Append("    .param .u32 p_").Append(CtcScalars[i]).Append(i + 1 < CtcScalars.Length ? ",\n" : "");
        }

        p.AppendLine();
        p.AppendLine(")");
        p.AppendLine("{");
        p.AppendLine("    .reg .pred %p<20>;");
        p.AppendLine("    .reg .f32 %f<48>;");
        p.AppendLine("    .reg .f64 %d<8>;");
        p.AppendLine("    .reg .b32 %r<64>;");
        p.AppendLine("    .reg .b64 %rd<16>;");
        p.Append("    .reg .u32 ");
        for (int i = 0; i < CtcScalars.Length; i++)
        {
            p.Append(i > 0 ? ", %s_" : "%s_").Append(CtcScalars[i]);
        }

        p.AppendLine(";");
        p.AppendLine("    .reg .u32 %n, %len, %labels, %offset, %S, %lane, %nt, %stride, %t, %wlane, %warp;");
        p.Append("    .reg .u64 %rows");
        foreach (string q in pointers)
        {
            p.Append(", %g_").Append(q);
        }

        p.AppendLine(";");
        p.AppendLine(CultureInfo.InvariantCulture, $"    .shared .align 4 .f32 red[{threads}];");
        p.AppendLine(CultureInfo.InvariantCulture, $"    .shared .align 4 .f32 wmax[{2 * (threads / 32)}];");
        p.AppendLine(CultureInfo.InvariantCulture, $"    .shared .align 4 .f32 wtop[{2 * (threads / 32)}];");
        foreach (string s in CtcScalars)
        {
            p.AppendLine($"    ld.param.u32 %s_{s}, [p_{s}];");
        }

        foreach (string q in pointers)
        {
            p.AppendLine($"    ld.param.u64 %g_{q}, [p_{q}];");
            p.AppendLine($"    cvta.to.global.u64 %g_{q}, %g_{q};");
        }

        p.AppendLine($"""
                mov.u32 %n, %ctaid.x;
                mov.u32 %lane, %tid.x;
                mov.u32 %nt, %ntid.x;
                and.b32 %wlane, %lane, 31;
                shr.u32 %warp, %lane, 5;
                setp.ge.u32 %p0, %n, %s_batch;
                @%p0 bra DONE;
                mul.lo.u32 %r1, %n, 12;
                cvt.u64.u32 %rd1, %r1;
                add.u64 %rd1, %g_meta, %rd1;
                ld.global.u32 %len, [%rd1];
                ld.global.u32 %labels, [%rd1+4];
                ld.global.u32 %offset, [%rd1+8];
                shl.b32 %S, %labels, 1;
                add.u32 %S, %S, 1;
            """);
        return p;
    }

    // The address of row element `index` (a register) into %rd2: the shared rows or the scratch rows at `rows` (%rows,
    // stride states).
    private static string RowAddress(bool shared, string index, string rows = "%rows") => shared
        ? $"""
            mov.u64 %rd2, ctc_rows;
            mul.wide.u32 %rd3, {index}, 4;
            add.u64 %rd2, %rd2, %rd3;
            """
        : $"""
            mul.wide.u32 %rd3, {index}, 4;
            add.u64 %rd2, {rows}, %rd3;
            """;

    private static string Space(bool shared) => shared ? "shared" : "global";

    // The log-probability of class `c` at step `t` into `target` (uses %r50, %rd4).
    private static string LogProb(string t, string c, string target) => $"""
        setp.ne.u32 %p10, %s_batchfirst, 0;
        mad.lo.u32 %r50, %n, %s_steps, {t};
        @!%p10 mad.lo.u32 %r50, {t}, %s_batch, %n;
        mad.lo.u32 %r50, %r50, %s_classes, {c};
        mul.wide.u32 %rd4, %r50, 4;
        add.u64 %rd4, %g_logprobs, %rd4;
        ld.global.f32 {target}, [%rd4];
        """;

    // The class of state `s` into `target`: the blank, or (odd s) label (s - 1) / 2 clamped to the classes (uses %r51, %r52, %f39, %rd5).
    private static string StateClass(string s, string target, string label) => $"""
        mov.u32 {target}, %s_blank;
        and.b32 %r51, {s}, 1;
        setp.eq.u32 %p11, %r51, 0;
        @%p11 bra {label}_CLASS;
        shr.u32 %r51, {s}, 1;
        {TargetLabel("%r51", target)}
        {label}_CLASS:
        """;

    // Label `j` of the sequence into `target`, clamped to the classes (uses %r52, %f39, %rd5).
    private static string TargetLabel(string j, string target) => $"""
        add.u32 %r52, %offset, {j};
        mul.wide.u32 %rd5, %r52, 4;
        add.u64 %rd5, %g_targets, %rd5;
        ld.global.f32 %f39, [%rd5];
        cvt.rzi.s32.f32 {target}, %f39;
        max.s32 {target}, {target}, 0;
        sub.u32 %r52, %s_classes, 1;
        min.s32 {target}, {target}, %r52;
        """;

    // `target` = log(e^a + e^b + e^c) with one logarithm, -∞ when all three are (uses %f30 … %f34, %p12).
    private static string LogSum3(string a, string b, string c, string target, string label) => $"""
        max.f32 %f30, {a}, {b};
        max.f32 %f30, %f30, {c};
        mov.f32 {target}, {NegInf};
        setp.eq.f32 %p12, %f30, {NegInf};
        @%p12 bra {label}_LOGSUM;
        sub.f32 %f31, {a}, %f30;
        mul.f32 %f31, %f31, {Log2E};
        ex2.approx.ftz.f32 %f31, %f31;
        sub.f32 %f32, {b}, %f30;
        mul.f32 %f32, %f32, {Log2E};
        ex2.approx.ftz.f32 %f32, %f32;
        add.f32 %f31, %f31, %f32;
        sub.f32 %f33, {c}, %f30;
        mul.f32 %f33, %f33, {Log2E};
        ex2.approx.ftz.f32 %f33, %f33;
        add.f32 %f31, %f31, %f33;
        lg2.approx.ftz.f32 %f34, %f31;
        fma.rn.f32 {target}, %f34, {Ln2}, %f30;
        {label}_LOGSUM:
        """;

    // e^x into `target` (uses %f35).
    private static string Exp(string x, string target) => $"""
        mul.f32 %f35, {x}, {Log2E};
        ex2.approx.ftz.f32 {target}, %f35;
        """;

    // α of step %t at state `s` (a register) from the row at element offset `from` into %f1, less the previous row's
    // maximum `shift`: the shared or scratch row, or (AlphaStepAlpha, `rows` %g_alpha) the alpha buffer row. Uses %r40 …
    // %r45, %f2 … %f6.
    private static string AlphaStep(bool shared, string s, string from, string shift, string label, string rows = "%rows") => $"""
        {StateClass(s, "%r40", label + "_C")}
        add.u32 %r41, {from}, {s};
        {RowAddress(shared, "%r41", rows)}
        ld.{Space(shared)}.f32 %f2, [%rd2];
        mov.f32 %f3, {NegInf};
        setp.eq.u32 %p13, {s}, 0;
        @%p13 bra {label}_NOB;
        ld.{Space(shared)}.f32 %f3, [%rd2+-4];
        {label}_NOB:
        mov.f32 %f4, {NegInf};
        setp.lt.u32 %p13, {s}, 2;
        @%p13 bra {label}_NOSKIP;
        setp.eq.u32 %p13, %r40, %s_blank;
        @%p13 bra {label}_NOSKIP;
        sub.u32 %r42, {s}, 2;
        {StateClass("%r42", "%r43", label + "_C2")}
        setp.eq.u32 %p13, %r43, %r40;
        @%p13 bra {label}_NOSKIP;
        add.u32 %r41, {from}, {s};
        {RowAddress(shared, "%r41", rows)}
        ld.{Space(shared)}.f32 %f4, [%rd2+-8];
        {label}_NOSKIP:
        {LogSum3("%f2", "%f3", "%f4", "%f5", label + "_LS")}
        mov.f32 %f1, {NegInf};
        setp.eq.f32 %p13, %f5, {NegInf};
        @%p13 bra {label}_DONE;
        {LogProb("%t", "%r40", "%f6")}
        sub.f32 %f5, %f5, {shift};
        add.f32 %f1, %f5, %f6;
        {label}_DONE:
        """;

    // α of step 0 at state `s` into %f1 (uses %r40).
    private static string AlphaStart(string s, string label) => $"""
        mov.f32 %f1, {NegInf};
        setp.ge.u32 %p13, {s}, 2;
        @%p13 bra {label}_START;
        {StateClass(s, "%r40", label + "_C")}
        mov.u32 %r44, 0;
        {LogProb("%r44", "%r40", "%f1")}
        {label}_START:
        """;

    // A loop of the block's threads over the states: `body` with the state in %r30.
    private static string StateLoop(string label, string body) => $"""
        mov.u32 %r30, %lane;
        {label}_LOOP:
        setp.ge.u32 %p14, %r30, %S;
        @%p14 bra {label}_END;
        {body}
        add.u32 %r30, %r30, %nt;
        bra {label}_LOOP;
        {label}_END:
        """;

    // The block's maximum of `value` into `target`, the same in every thread, 0 when it is -∞ (a row's offset; with
    // `keepInfinity` it stays -∞): the warp's lanes by shuffles, then each warp's maximum in its slot of `slots` and the
    // slots by shuffles again. One barrier, which also stands for the step's barrier; the slots alternate halves with the
    // parity of %t, so the next step's writes never meet this step's reads. No branches: every thread runs it. Uses %f46,
    // %r55, %r56, %rd12, %rd13, %p16; `value` changes.
    private static string BlockMax(string value, string target, string slots = "wmax", bool keepInfinity = false)
    {
        int warps = BlockSize / 32;
        var s = new StringBuilder();
        foreach (int offset in new[] { 16, 8, 4, 2, 1 })
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"shfl.sync.bfly.b32 %f46, {value}, {offset}, 31, 0xffffffff;");
            s.AppendLine($"max.f32 {value}, {value}, %f46;");
        }

        s.AppendLine(CultureInfo.InvariantCulture, $"""
            and.b32 %r55, %t, 1;
            mul.lo.u32 %r55, %r55, {warps};
            add.u32 %r56, %r55, %warp;
            mov.u64 %rd12, {slots};
            mul.wide.u32 %rd13, %r56, 4;
            add.u64 %rd13, %rd12, %rd13;
            setp.eq.u32 %p16, %wlane, 0;
            @%p16 st.shared.f32 [%rd13], {value};
            bar.sync 0;
            and.b32 %r56, %wlane, {warps - 1};
            add.u32 %r56, %r56, %r55;
            mul.wide.u32 %rd13, %r56, 4;
            add.u64 %rd13, %rd12, %rd13;
            ld.shared.f32 {target}, [%rd13];
            """);
        for (int offset = warps / 2; offset >= 1; offset /= 2)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"shfl.sync.bfly.b32 %f46, {target}, {offset}, 31, 0xffffffff;");
            s.AppendLine($"max.f32 {target}, {target}, %f46;");
        }

        if (!keepInfinity)
        {
            s.AppendLine($"setp.eq.f32 %p16, {target}, {NegInf};");
            s.Append($"@%p16 mov.f32 {target}, {Zero};");
        }

        return s.ToString();
    }

    // The block's reduction of `value` (max or sum) into `target`, the same in every thread (a tree over shared `red`).
    private static string BlockReduce(string value, string target, bool max, string label) => $"""
        mov.u64 %rd6, red;
        mul.wide.u32 %rd7, %lane, 4;
        add.u64 %rd7, %rd6, %rd7;
        st.shared.f32 [%rd7], {value};
        bar.sync 0;
        shr.u32 %r53, %nt, 1;
        {label}_TREE:
        setp.eq.u32 %p15, %r53, 0;
        @%p15 bra {label}_TREE_END;
        setp.ge.u32 %p15, %lane, %r53;
        @%p15 bra {label}_TREE_WAIT;
        mul.wide.u32 %rd8, %r53, 4;
        add.u64 %rd8, %rd7, %rd8;
        ld.shared.f32 %f36, [%rd8];
        ld.shared.f32 %f37, [%rd7];
        {(max ? "max.f32" : "add.f32")} %f37, %f37, %f36;
        st.shared.f32 [%rd7], %f37;
        {label}_TREE_WAIT:
        bar.sync 0;
        shr.u32 %r53, %r53, 1;
        bra {label}_TREE;
        {label}_TREE_END:
        ld.shared.f32 {target}, [%rd6];
        bar.sync 0;
        """;

    // losses[n] = -log p(labels | sequence n): two rows of α, alternating.
    private static void CtcLossKernel(StringBuilder sb, bool shared)
    {
        string name = shared ? "ctc_loss_f32" : "ctc_loss_global_f32";
        var p = CtcHeader(name, ["logprobs", "targets", "meta", "losses", "work"]);
        p.AppendLine("    mov.u32 %stride, %s_states;");
        p.AppendLine(shared
            ? ""
            : """
                  mul.lo.u32 %r1, %n, %s_states;
                  shl.b32 %r1, %r1, 1;
                  mul.wide.u32 %rd1, %r1, 4;
                  add.u64 %rows, %g_work, %rd1;
              """);
        p.AppendLine($"""
                mul.wide.u32 %rd9, %n, 4;
                add.u64 %rd9, %g_losses, %rd9;
                setp.ne.u32 %p1, %len, 0;
                @%p1 bra RUN;
                setp.ne.u32 %p0, %lane, 0;
                @%p0 bra DONE;
                mov.f32 %f20, 0f7F800000;
                setp.eq.u32 %p2, %labels, 0;
                setp.ne.or.u32 %p2, %s_zeroinf, 0, %p2;
                @%p2 mov.f32 %f20, {Zero};
                st.global.f32 [%rd9], %f20;
                bra DONE;
            RUN:
            """);
        // %f40: the thread's largest of the row; %f41: the row's maximum (the next step subtracts it); %d0: the offset P_t.
        p.AppendLine($"    mov.f32 %f40, {NegInf};");
        p.AppendLine(StateLoop("INIT", $"""
            {AlphaStart("%r30", "I")}
            {RowAddress(shared, "%r30")}
            st.{Space(shared)}.f32 [%rd2], %f1;
            max.f32 %f40, %f40, %f1;
            """));
        p.AppendLine("    mov.u32 %t, 0;");
        p.AppendLine(BlockMax("%f40", "%f41"));
        p.AppendLine($$"""
                mov.f64 %d0, 0d0000000000000000;
                mov.u32 %t, 1;
            STEP:
                setp.ge.u32 %p1, %t, %len;
                @%p1 bra STEPS_END;
                cvt.f64.f32 %d1, %f41;
                add.f64 %d0, %d0, %d1;
                sub.u32 %r20, %t, 1;
                and.b32 %r20, %r20, 1;
                mul.lo.u32 %r20, %r20, %stride;
                and.b32 %r21, %t, 1;
                mul.lo.u32 %r21, %r21, %stride;
                mov.f32 %f40, {{NegInf}};
                {{StateLoop("ADV", $"""
                    {AlphaStep(shared, "%r30", "%r20", "%f41", "A")}
                    add.u32 %r41, %r21, %r30;
                    {RowAddress(shared, "%r41")}
                    st.{Space(shared)}.f32 [%rd2], %f1;
                    max.f32 %f40, %f40, %f1;
                    """)}}
                {{BlockMax("%f40", "%f41")}}
                add.u32 %t, %t, 1;
                bra STEP;
            STEPS_END:
                setp.ne.u32 %p0, %lane, 0;
                @%p0 bra DONE;
                sub.u32 %r20, %len, 1;
                and.b32 %r20, %r20, 1;
                mul.lo.u32 %r20, %r20, %stride;
                add.u32 %r41, %r20, %S;
                sub.u32 %r41, %r41, 1;
                {{RowAddress(shared, "%r41")}}
                ld.{{Space(shared)}}.f32 %f10, [%rd2];
                mov.f32 %f11, {{NegInf}};
                setp.lt.u32 %p2, %S, 2;
                @!%p2 ld.{{Space(shared)}}.f32 %f11, [%rd2+-4];
                mov.f32 %f12, {{NegInf}};
                {{LogSum3("%f10", "%f11", "%f12", "%f13", "END")}}
                cvt.f64.f32 %d1, %f13;
                add.f64 %d1, %d1, %d0;
                neg.f64 %d1, %d1;
                cvt.rn.f32.f64 %f13, %d1;
                setp.eq.f32 %p3, %f13, 0f7F800000;
                setp.ne.and.u32 %p3, %s_zeroinf, 0, %p3;
                @%p3 mov.f32 %f13, {{Zero}};
                st.global.f32 [%rd9], %f13;
            DONE:
                ret;
            }

            """);
        sb.Append(p);
    }

    // dlogprobs += lossgrads[n] · ∂losses[n] / ∂logprobs: α of every step into `alpha`, then β backwards a row at a time
    // with each step's per-class sums of α·β (scaled by the step's largest).
    private static void CtcBackwardKernel(StringBuilder sb, bool shared)
    {
        string name = shared ? "ctc_loss_bwd_f32" : "ctc_loss_bwd_global_f32";
        var p = CtcHeader(name, ["logprobs", "targets", "meta", "lossgrads", "dlogprobs", "alpha", "work"]);

        // %rows: β's two rows, the terms, then each label's class (from element %r2) and its next occurrence of that class
        // (from %r3; the top bit set on all but a class's first occurrence) (shared, or the scratch); %r9: the alpha rows'
        // first element of this sequence; %r10: its row maxima's first; %f21: the scale; %f22: the end states' log-sum of
        // the last α row (the nll is -(that + P_{T-1})); %f41: α's row maximum, %f43 β's; %d2: R_t; %d3: P_{T-1} - P_t;
        // %f45: top + nll of step t in float, from those in double.
        p.AppendLine("    mov.u32 %stride, %s_states;");
        p.AppendLine(shared
            ? ""
            : string.Create(CultureInfo.InvariantCulture, $"""
                  mul.lo.u32 %r1, %n, %s_states;
                  mul.lo.u32 %r1, %r1, {CtcRows(backward: true)};
                  mul.wide.u32 %rd1, %r1, 4;
                  add.u64 %rows, %g_work, %rd1;
              """));
        p.AppendLine($$"""
                mul.wide.u32 %rd9, %n, 4;
                add.u64 %rd9, %g_lossgrads, %rd9;
                ld.global.f32 %f21, [%rd9];
                setp.eq.f32 %p1, %f21, {{Zero}};
                @%p1 bra DONE;
                setp.eq.u32 %p1, %len, 0;
                @%p1 bra DONE;
                {{LabelGroups(shared)}}
                mul.lo.u32 %r9, %n, %s_steps;
                mul.lo.u32 %r9, %r9, %s_states;
                mul.lo.u32 %r10, %s_batch, %s_steps;
                mul.lo.u32 %r10, %r10, %s_states;
                mad.lo.u32 %r10, %n, %s_steps, %r10;
                setp.eq.u32 %p17, %lane, 0;
                mov.f32 %f40, {{NegInf}};
                {{StateLoop("INIT", $"""
                    {AlphaStart("%r30", "I")}
                    add.u32 %r41, %r9, %r30;
                    mul.wide.u32 %rd10, %r41, 4;
                    add.u64 %rd10, %g_alpha, %rd10;
                    st.global.f32 [%rd10], %f1;
                    max.f32 %f40, %f40, %f1;
                    """)}}
                mov.u32 %t, 0;
                {{BlockMax("%f40", "%f41")}}
                {{RowMaximum("st", "%f41")}}
                mov.u32 %t, 1;
            ASTEP:
                setp.ge.u32 %p1, %t, %len;
                @%p1 bra ASTEPS_END;
                sub.u32 %r20, %t, 1;
                mad.lo.u32 %r20, %r20, %s_states, %r9;
                mad.lo.u32 %r21, %t, %s_states, %r9;
                mov.f32 %f40, {{NegInf}};
                {{StateLoop("ADV", $"""
                    {AlphaStepAlpha("%r30", "%r20", "%f41", "A")}
                    add.u32 %r41, %r21, %r30;
                    mul.wide.u32 %rd10, %r41, 4;
                    add.u64 %rd10, %g_alpha, %rd10;
                    st.global.f32 [%rd10], %f1;
                    max.f32 %f40, %f40, %f1;
                    """)}}
                {{BlockMax("%f40", "%f41")}}
                {{RowMaximum("st", "%f41")}}
                add.u32 %t, %t, 1;
                bra ASTEP;
            ASTEPS_END:
                sub.u32 %r20, %len, 1;
                mad.lo.u32 %r20, %r20, %s_states, %r9;
                add.u32 %r41, %r20, %S;
                sub.u32 %r41, %r41, 1;
                mul.wide.u32 %rd10, %r41, 4;
                add.u64 %rd10, %g_alpha, %rd10;
                ld.global.f32 %f10, [%rd10];
                mov.f32 %f11, {{NegInf}};
                setp.lt.u32 %p2, %S, 2;
                @!%p2 ld.global.f32 %f11, [%rd10+-4];
                mov.f32 %f12, {{NegInf}};
                {{LogSum3("%f10", "%f11", "%f12", "%f22", "NLL")}}
                setp.eq.f32 %p3, %f22, {{NegInf}};
                setp.ne.and.u32 %p3, %s_zeroinf, 0, %p3;
                @%p3 bra DONE;
                bar.sync 0;
                mov.f64 %d2, 0d0000000000000000;
                mov.f64 %d3, 0d0000000000000000;
                mov.f32 %f43, {{Zero}};
                sub.u32 %t, %len, 1;
            BSTEP:
                setp.lt.s32 %p1, %t, 0;
                @%p1 bra DONE;
                and.b32 %r22, %t, 1;
                mul.lo.u32 %r22, %r22, %stride;
                add.u32 %r23, %t, 1;
                and.b32 %r23, %r23, 1;
                mul.lo.u32 %r23, %r23, %stride;
                sub.u32 %r24, %len, 1;
                setp.eq.u32 %p4, %t, %r24;
                @%p4 bra OFFSETS_END;
                cvt.f64.f32 %d1, %f43;
                add.f64 %d2, %d2, %d1;
                {{RowMaximum("ld", "%f47")}}
                cvt.f64.f32 %d1, %f47;
                add.f64 %d3, %d3, %d1;
            OFFSETS_END:
                mov.f32 %f44, {{NegInf}};
                {{StateLoop("BETA", BetaState(shared) + "\n    max.f32 %f44, %f44, %f7;")}}
                {{BlockMax("%f44", "%f43")}}
                mad.lo.u32 %r25, %t, %s_states, %r9;
                mov.f32 %f23, {{NegInf}};
                {{StateLoop("TOP", $"""
                    add.u32 %r41, %r25, %r30;
                    mul.wide.u32 %rd10, %r41, 4;
                    add.u64 %rd10, %g_alpha, %rd10;
                    ld.global.f32 %f24, [%rd10];
                    add.u32 %r41, %r22, %r30;
                    {RowAddress(shared, "%r41")}
                    ld.{Space(shared)}.f32 %f25, [%rd2];
                    add.f32 %f24, %f24, %f25;
                    max.f32 %f23, %f23, %f24;
                    """)}}
                {{BlockMax("%f23", "%f26", "wtop", keepInfinity: true)}}
                cvt.f64.f32 %d4, %f26;
                cvt.f64.f32 %d1, %f22;
                sub.f64 %d4, %d4, %d1;
                add.f64 %d4, %d4, %d2;
                sub.f64 %d4, %d4, %d3;
                cvt.rn.f32.f64 %f45, %d4;
                setp.eq.f32 %p5, %f26, {{NegInf}};
                shl.b32 %r26, %stride, 1;
                {{StateLoop("TERM", $"""
                    add.u32 %r41, %r25, %r30;
                    mul.wide.u32 %rd10, %r41, 4;
                    add.u64 %rd10, %g_alpha, %rd10;
                    ld.global.f32 %f24, [%rd10];
                    add.u32 %r41, %r22, %r30;
                    {RowAddress(shared, "%r41")}
                    ld.{Space(shared)}.f32 %f25, [%rd2];
                    add.f32 %f24, %f24, %f25;
                    sub.f32 %f24, %f24, %f26;
                    {Exp("%f24", "%f24")}
                    @%p5 mov.f32 %f24, {Zero};
                    add.u32 %r41, %r26, %r30;
                    {RowAddress(shared, "%r41")}
                    st.{Space(shared)}.f32 [%rd2], %f24;
                    """)}}
                bar.sync 0;
                mov.f32 %f27, {{Zero}};
                shl.b32 %r30, %lane, 1;
                shl.b32 %r27, %nt, 1;
            BLANK:
                setp.ge.u32 %p6, %r30, %S;
                @%p6 bra BLANK_END;
                add.u32 %r41, %r26, %r30;
                {{RowAddress(shared, "%r41")}}
                ld.{{Space(shared)}}.f32 %f24, [%rd2];
                add.f32 %f27, %f27, %f24;
                add.u32 %r30, %r30, %r27;
                bra BLANK;
            BLANK_END:
                {{BlockReduce("%f27", "%f28", max: false, "RSUM")}}
                setp.ne.u32 %p6, %lane, 0;
                @%p6 bra LABELS;
                {{ClassGradient("%s_blank", "%f28", "GB")}}
            LABELS:
                mov.u32 %r31, %lane;
            LABEL:
                setp.ge.u32 %p7, %r31, %labels;
                @%p7 bra LABELS_END;
                add.u32 %r41, %r3, %r31;
                {{RowAddress(shared, "%r41")}}
                ld.{{Space(shared)}}.u32 %r33, [%rd2];
                and.b32 %r34, %r33, 0x80000000;
                setp.ne.u32 %p8, %r34, 0;
                @%p8 bra LABEL_NEXT;
                add.u32 %r41, %r2, %r31;
                {{RowAddress(shared, "%r41")}}
                ld.{{Space(shared)}}.u32 %r32, [%rd2];
                mov.f32 %f29, {{Zero}};
                mov.u32 %r37, %r31;
            OCCURRENCE:
                shl.b32 %r35, %r37, 1;
                add.u32 %r35, %r35, 1;
                add.u32 %r41, %r26, %r35;
                {{RowAddress(shared, "%r41")}}
                ld.{{Space(shared)}}.f32 %f24, [%rd2];
                add.f32 %f29, %f29, %f24;
                add.u32 %r41, %r3, %r37;
                {{RowAddress(shared, "%r41")}}
                ld.{{Space(shared)}}.u32 %r37, [%rd2];
                and.b32 %r37, %r37, 0x7FFFFFFF;
                setp.lt.u32 %p8, %r37, %labels;
                @%p8 bra OCCURRENCE;
                {{ClassGradient("%r32", "%f29", "GL")}}
            LABEL_NEXT:
                add.u32 %r31, %r31, %nt;
                bra LABEL;
            LABELS_END:
                bar.sync 0;
                sub.u32 %t, %t, 1;
                bra BSTEP;
            DONE:
                ret;
            }

            """);
        sb.Append(p);
    }

    // The label groups, once per call: each label's class at element %r2 = 3 · states of the rows, then its next occurrence
    // of the same class (or the label count) at %r3 = %r2 + labels, the top bit set when an earlier label has the class.
    // Uses %r31 … %r36, %r41, %p18 (and TargetLabel's, RowAddress's registers).
    private static string LabelGroups(bool shared) => $"""
        mul.lo.u32 %r2, %stride, 3;
        add.u32 %r3, %r2, %labels;
        mov.u32 %r31, %lane;
        GCLASS:
        setp.ge.u32 %p18, %r31, %labels;
        @%p18 bra GCLASS_END;
        {TargetLabel("%r31", "%r32")}
        add.u32 %r41, %r2, %r31;
        {RowAddress(shared, "%r41")}
        st.{Space(shared)}.u32 [%rd2], %r32;
        add.u32 %r31, %r31, %nt;
        bra GCLASS;
        GCLASS_END:
        bar.sync 0;
        mov.u32 %r31, %lane;
        GNEXT:
        setp.ge.u32 %p18, %r31, %labels;
        @%p18 bra GNEXT_END;
        add.u32 %r41, %r2, %r31;
        {RowAddress(shared, "%r41")}
        ld.{Space(shared)}.u32 %r32, [%rd2];
        mov.u32 %r36, 0;
        mov.u32 %r33, 0;
        GEARLIER:
        setp.ge.u32 %p18, %r33, %r31;
        @%p18 bra GEARLIER_END;
        add.u32 %r41, %r2, %r33;
        {RowAddress(shared, "%r41")}
        ld.{Space(shared)}.u32 %r34, [%rd2];
        setp.eq.u32 %p18, %r34, %r32;
        @%p18 mov.u32 %r36, 0x80000000;
        @%p18 bra GEARLIER_END;
        add.u32 %r33, %r33, 1;
        bra GEARLIER;
        GEARLIER_END:
        add.u32 %r33, %r31, 1;
        GLATER:
        setp.ge.u32 %p18, %r33, %labels;
        @%p18 bra GLATER_END;
        add.u32 %r41, %r2, %r33;
        {RowAddress(shared, "%r41")}
        ld.{Space(shared)}.u32 %r34, [%rd2];
        setp.eq.u32 %p18, %r34, %r32;
        @%p18 bra GLATER_END;
        add.u32 %r33, %r33, 1;
        bra GLATER;
        GLATER_END:
        or.b32 %r33, %r33, %r36;
        add.u32 %r41, %r3, %r31;
        {RowAddress(shared, "%r41")}
        st.{Space(shared)}.u32 [%rd2], %r33;
        add.u32 %r31, %r31, %nt;
        bra GNEXT;
        GNEXT_END:
        bar.sync 0;
        """;

    // α of step %t at state `s` from the alpha buffer's row at element `from` (the gradient keeps every row there) into %f1.
    private static string AlphaStepAlpha(string s, string from, string shift, string label) =>
        AlphaStep(false, s, from, shift, label, rows: "%g_alpha");

    // Step %t's α row maximum m_t, kept after all the α rows (element %r10 + t of the alpha buffer, %r10 = batch · steps ·
    // states + n · steps): "st" writes `reg` there (thread 0, %p17), "ld" reads it into `reg` (every thread). Uses %r54, %rd11.
    private static string RowMaximum(string op, string reg) => $"""
        add.u32 %r54, %r10, %t;
        mul.wide.u32 %rd11, %r54, 4;
        add.u64 %rd11, %g_alpha, %rd11;
        {(op == "st" ? $"@%p17 st.global.f32 [%rd11], {reg};" : $"ld.global.f32 {reg}, [%rd11];")}
        """;

    // β of step %t at state %r30 into row %r22 from row %r23 (%p4: the last step), less the later row's maximum %f43; the
    // value stays in %f7. Uses %r40 … %r46, %f2 … %f8.
    private static string BetaState(bool shared) => $"""
        {StateClass("%r30", "%r40", "B_C")}
        mov.f32 %f7, {NegInf};
        @!%p4 bra B_RECUR;
        sub.u32 %r45, %S, 2;
        setp.lt.s32 %p13, %r30, %r45;
        @%p13 bra B_STORE;
        {LogProb("%t", "%r40", "%f7")}
        bra B_STORE;
        B_RECUR:
        add.u32 %r41, %r23, %r30;
        {RowAddress(shared, "%r41")}
        ld.{Space(shared)}.f32 %f2, [%rd2];
        mov.f32 %f3, {NegInf};
        add.u32 %r46, %r30, 1;
        setp.ge.u32 %p13, %r46, %S;
        @%p13 bra B_NOB;
        ld.{Space(shared)}.f32 %f3, [%rd2+4];
        B_NOB:
        mov.f32 %f4, {NegInf};
        add.u32 %r46, %r30, 2;
        setp.ge.u32 %p13, %r46, %S;
        @%p13 bra B_NOSKIP;
        setp.eq.u32 %p13, %r40, %s_blank;
        @%p13 bra B_NOSKIP;
        {StateClass("%r46", "%r43", "B_C2")}
        setp.eq.u32 %p13, %r43, %r40;
        @%p13 bra B_NOSKIP;
        add.u32 %r41, %r23, %r30;
        {RowAddress(shared, "%r41")}
        ld.{Space(shared)}.f32 %f4, [%rd2+8];
        B_NOSKIP:
        {LogSum3("%f2", "%f3", "%f4", "%f5", "B_LS")}
        setp.eq.f32 %p13, %f5, {NegInf};
        @%p13 bra B_STORE;
        {LogProb("%t", "%r40", "%f6")}
        sub.f32 %f5, %f5, %f43;
        add.f32 %f7, %f5, %f6;
        B_STORE:
        add.u32 %r41, %r22, %r30;
        {RowAddress(shared, "%r41")}
        st.{Space(shared)}.f32 [%rd2], %f7;
        """;

    // dlogprobs[t, class] -= scale · sum · e^(top + nll - logprobs[t, class]) (scale %f21; top + nll %f45, the true α·β's
    // with the offsets put back). Uses %r50, %rd4, %f8, %f9.
    private static string ClassGradient(string c, string sum, string label) => $"""
        {LogProb("%t", c, "%f8")}
        sub.f32 %f9, %f45, %f8;
        {Exp("%f9", "%f9")}
        mul.f32 %f9, %f9, {sum};
        mul.f32 %f9, %f9, %f21;
        sub.u64 %rd4, %rd4, %g_logprobs;
        add.u64 %rd4, %rd4, %g_dlogprobs;
        ld.global.f32 %f8, [%rd4];
        sub.f32 %f8, %f8, %f9;
        st.global.f32 [%rd4], %f8;
        """;
}
