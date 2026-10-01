// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;

namespace Idrak.Backends.Cuda;

// PTX for classification, normalization, embeddings, convolution, pooling and N-D shape operations.
internal static partial class PtxKernels
{
    public static readonly string[] AdvancedNames =
    [
        "exp_f32", "exp_bwd_f32", "log_f32", "log_bwd_f32", "gelu_f32", "gelu_bwd_f32", "inv_sqrt_f32",
        "softmax_f32", "softmax_bwd_f32", "argmax_f32", "class_match_f32",
        "norm_stats_f32", "norm_apply_f32", "norm_bwd_f32", "group_scale_shift_f32", "group_reduce_f32",
        "gather_f32", "gather_bf16_f32", "one_hot_f32", "scatter_add_f32", "im2col_f32", "col2im_f32", "maxpool_f32", "maxpool_bwd_f32",
        "permute_f32", "copy2d_f32", "sum_axis_f32", "broadcast_axis_f32",
    ];

    public const int MaxPermuteRank = 6;

    // Constants (not static readonly): Source is built during static initialization, before other partial files'
    // static fields are guaranteed to be initialized.
    private const string Log2E = "0f3FB8AA3B";
    private const string Ln2 = "0f3F317218";
    private const string NegInf = "0fFF800000";

    private static void BuildAdvanced(StringBuilder sb)
    {
        ElementWiseMath(sb);
        RowKernels(sb);
        Normalization(sb);
        Embedding(sb);
        Convolution(sb);
        ShapeKernels(sb);
    }

    // ------------------------------------------------------------------ element-wise math

    private static void ElementWiseMath(StringBuilder sb)
    {
        Elementwise(sb, "exp_f32", ["x", "y"], [],
            $"""
            ld.global.f32 %f1, [%a_x];
            mul.f32 %f2, %f1, {Log2E};
            ex2.approx.ftz.f32 %f3, %f2;
            st.global.f32 [%a_y], %f3;
            """);

        Elementwise(sb, "exp_bwd_f32", ["x", "y", "dy", "dx"], [],
            """
            ld.global.f32 %f1, [%a_y];
            ld.global.f32 %f2, [%a_dy];
            ld.global.f32 %f3, [%a_dx];
            fma.rn.f32 %f3, %f2, %f1, %f3;
            st.global.f32 [%a_dx], %f3;
            """);

        Elementwise(sb, "log_f32", ["x", "y"], [],
            $"""
            ld.global.f32 %f1, [%a_x];
            lg2.approx.ftz.f32 %f2, %f1;
            mul.f32 %f3, %f2, {Ln2};
            st.global.f32 [%a_y], %f3;
            """);

        Elementwise(sb, "log_bwd_f32", ["x", "y", "dy", "dx"], [],
            """
            ld.global.f32 %f1, [%a_x];
            ld.global.f32 %f2, [%a_dy];
            ld.global.f32 %f3, [%a_dx];
            div.rn.f32 %f4, %f2, %f1;
            add.f32 %f3, %f3, %f4;
            st.global.f32 [%a_dx], %f3;
            """);

        // GELU (tanh approximation). %f6 = t = tanh(k (x + c x³)).
        string geluTanh = $"""
            ld.global.f32 %f1, [%a_x];
            mul.f32 %f2, %f1, %f1;
            mul.f32 %f3, %f2, %f1;
            fma.rn.f32 %f4, %f3, {F(0.044715f)}, %f1;
            mul.f32 %f4, %f4, {F(0.7978845608f)};
            mul.f32 %f5, %f4, {F(2.8853900817779268f)};
            ex2.approx.ftz.f32 %f5, %f5;
            add.f32 %f5, %f5, {One};
            rcp.rn.f32 %f5, %f5;
            fma.rn.f32 %f6, %f5, {F(-2f)}, {One};
            """;

        Elementwise(sb, "gelu_f32", ["x", "y"], [],
            geluTanh + $"""

            add.f32 %f7, %f6, {One};
            mul.f32 %f7, %f7, %f1;
            mul.f32 %f7, %f7, {F(0.5f)};
            st.global.f32 [%a_y], %f7;
            """);

        Elementwise(sb, "gelu_bwd_f32", ["x", "y", "dy", "dx"], [],
            geluTanh + $"""

            add.f32 %f7, %f6, {One};
            mul.f32 %f7, %f7, {F(0.5f)};
            mul.f32 %f8, %f6, %f6;
            sub.f32 %f8, {One}, %f8;
            mul.f32 %f9, %f2, {F(3f * 0.044715f)};
            add.f32 %f9, %f9, {One};
            mul.f32 %f9, %f9, {F(0.7978845608f)};
            mul.f32 %f8, %f8, %f9;
            mul.f32 %f8, %f8, %f1;
            fma.rn.f32 %f7, %f8, {F(0.5f)}, %f7;
            ld.global.f32 %f10, [%a_dy];
            ld.global.f32 %f11, [%a_dx];
            fma.rn.f32 %f11, %f10, %f7, %f11;
            st.global.f32 [%a_dx], %f11;
            """);

        Elementwise(sb, "inv_sqrt_f32", ["x", "y"], [("f32", "eps")],
            """
            ld.global.f32 %f1, [%a_x];
            add.f32 %f2, %f1, %s_eps;
            sqrt.rn.f32 %f3, %f2;
            rcp.rn.f32 %f4, %f3;
            st.global.f32 [%a_y], %f4;
            """);
    }

    // ------------------------------------------------------------------ one block per row

    private static void RowKernels(StringBuilder sb)
    {
        const string BlockRowStart = """
            mul.lo.u32 %r5, %row, %s_cols;
            mul.wide.u32 %rd1, %r5, 4;
            """;

        // Softmax / log-softmax, one block per row: max, then Σ exp(x - max) (exp stored in y for softmax; not for
        // log-softmax, which reads x again and may be writing over it), then normalize.
        // Each thread revisits only the columns it wrote, so the passes need no synchronization between them.
        RowBlock(sb, "softmax_f32", ["x", "y"], [("u32", "cols"), ("u32", "log")],
            BlockRowStart + $"""
            add.u64 %rd2, %b_x, %rd1;
            add.u64 %rd3, %b_y, %rd1;
            sub.u64 %rd6, %rd3, %rd2;
            mov.f32 %f1, {NegInf};
            """ + "\n" + StridedLoop("MAX", "%rd2", "%s_cols", "max.f32 %f1, %f1, %f2;") + "\n"
            + BlockReduce("RMAX", "%f1", "max", NegInf) + "\n" + $"""
            mov.f32 %f3, {Zero};
            setp.ne.u32 %p3, %s_log, 0;
            """ + "\n" + StridedLoop("EXP", "%rd2", "%s_cols", $"""
            sub.f32 %f2, %f2, %f1;
            mul.f32 %f2, %f2, {Log2E};
            ex2.approx.ftz.f32 %f2, %f2;
            add.u64 %rd7, %rd5, %rd6;
            @!%p3 st.global.f32 [%rd7], %f2;
            add.f32 %f3, %f3, %f2;
            """) + "\n" + BlockReduce("RSUM", "%f3", "add", Zero) + "\n" + $"""
            @%p3 bra LOGPATH;
            rcp.rn.f32 %f4, %f3;
            """ + "\n" + StridedLoop("NORM", "%rd3", "%s_cols", """
            mul.f32 %f2, %f2, %f4;
            st.global.f32 [%rd5], %f2;
            """) + "\n" + $"""
            bra FINISHED;
            LOGPATH:
            lg2.approx.ftz.f32 %f5, %f3;
            fma.rn.f32 %f5, %f5, {Ln2}, %f1;
            """ + "\n" + StridedLoop("LOGNORM", "%rd2", "%s_cols", """
            sub.f32 %f2, %f2, %f5;
            add.u64 %rd7, %rd5, %rd6;
            st.global.f32 [%rd7], %f2;
            """) + "\nFINISHED:");

        // Softmax: dx += y (dy - Σ dy·y). Log-softmax: dx += dy - exp(y) Σ dy. One block per row.
        RowBlock(sb, "softmax_bwd_f32", ["y", "dy", "dx"], [("u32", "cols"), ("u32", "log")],
            BlockRowStart + $"""
            add.u64 %rd2, %b_y, %rd1;
            add.u64 %rd3, %b_dy, %rd1;
            add.u64 %rd4, %b_dx, %rd1;
            sub.u64 %rd6, %rd3, %rd2;
            sub.u64 %rd8, %rd4, %rd2;
            setp.ne.u32 %p3, %s_log, 0;
            mov.f32 %f1, {Zero};
            """ + "\n" + StridedLoop("DOT", "%rd2", "%s_cols", """
            add.u64 %rd7, %rd5, %rd6;
            ld.global.f32 %f3, [%rd7];
            mul.f32 %f4, %f3, %f2;
            selp.f32 %f4, %f3, %f4, %p3;
            add.f32 %f1, %f1, %f4;
            """) + "\n" + BlockReduce("RDOT", "%f1", "add", Zero) + "\n" + StridedLoop("GRAD", "%rd2", "%s_cols", $"""
            add.u64 %rd7, %rd5, %rd6;
            ld.global.f32 %f3, [%rd7];
            add.u64 %rd9, %rd5, %rd8;
            ld.global.f32 %f4, [%rd9];
            sub.f32 %f5, %f3, %f1;
            mul.f32 %f5, %f5, %f2;
            mul.f32 %f6, %f2, {Log2E};
            ex2.approx.ftz.f32 %f6, %f6;
            mul.f32 %f6, %f6, %f1;
            sub.f32 %f6, %f3, %f6;
            selp.f32 %f5, %f6, %f5, %p3;
            add.f32 %f4, %f4, %f5;
            st.global.f32 [%rd9], %f4;
            """));

        // Index of the row's first maximum: each thread's first maximum over its columns, then the block's (ties go to the
        // smaller index). One block per row.
        static string RowArgMax(string label, string rowPtr, string value, string index) => $"""
            mov.f32 {value}, {NegInf};
            mov.u32 {index}, 0xFFFFFFFF;
            """ + "\n" + StridedLoop(label, rowPtr, "%s_cols", $"""
            setp.gt.f32 %p8, %f2, {value};
            setp.eq.u32 %p9, {index}, 0xFFFFFFFF;
            or.pred %p8, %p8, %p9;
            selp.f32 {value}, %f2, {value}, %p8;
            selp.b32 {index}, %r6, {index}, %p8;
            """) + "\n" + BlockArgMax("R" + label, value, index);

        RowBlock(sb, "argmax_f32", ["x", "y"], [("u32", "cols")],
            BlockRowStart + """
            add.u64 %rd2, %b_x, %rd1;
            """ + "\n" + RowArgMax("ARG", "%rd2", "%f1", "%r7") + "\n" + """
            setp.eq.u32 %p3, %tx, 0;
            cvt.rn.f32.u32 %f7, %r7;
            @%p3 st.global.f32 [%a_y], %f7;
            """);

        RowBlock(sb, "class_match_f32", ["p", "t", "y"], [("u32", "cols"), ("f32", "threshold")],
            BlockRowStart + $"""
            setp.eq.u32 %p3, %tx, 0;
            setp.ne.u32 %p4, %s_cols, 1;
            @%p4 bra MULTI;
            ld.global.f32 %f8, [%a_p];
            ld.global.f32 %f9, [%a_t];
            setp.ge.f32 %p5, %f8, %s_threshold;
            setp.ge.f32 %p6, %f9, {F(0.5f)};
            xor.pred %p5, %p5, %p6;
            selp.f32 %f10, {Zero}, {One}, %p5;
            bra WRITE;
            MULTI:
            add.u64 %rd2, %b_p, %rd1;
            add.u64 %rd3, %b_t, %rd1;
            """ + "\n" + RowArgMax("ARGP", "%rd2", "%f1", "%r7") + "\n" + RowArgMax("ARGT", "%rd3", "%f3", "%r8") + "\n" + $"""
            setp.eq.u32 %p5, %r7, %r8;
            selp.f32 %f10, {One}, {Zero}, %p5;
            WRITE:
            @%p3 st.global.f32 [%a_y], %f10;
            """);
    }

    // ------------------------------------------------------------------ grouped normalization

    /// <summary>Emits the index of group element %rJ: ((j / inner) * groups + g) * inner + j % inner, into %rIdx.</summary>
    private static string GroupElement(string j, string idx) => $"""
        div.u32 %r10, {j}, %inner;
        rem.u32 %r11, {j}, %inner;
        mad.lo.u32 %r12, %r10, %groups, %g;
        mad.lo.u32 {idx}, %r12, %inner, %r11;
        """;

    /// <summary>Block-wide sum of <paramref name="value"/> through shared array <paramref name="shared"/>; afterwards every thread can read [%sbase].</summary>
    private static string BlockReduce(string shared, string value, string label)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"mov.u32 %sbase, {shared};");
        sb.AppendLine("add.u32 %saddr, %sbase, %toff;");
        sb.AppendLine($"st.shared.f32 [%saddr], {value};");
        sb.AppendLine("bar.sync 0;");
        for (int stride = BlockSize / 2; stride > 0; stride /= 2)
        {
            sb.AppendLine($"setp.ge.u32 %p2, %thr, {stride};");
            sb.AppendLine($"@%p2 bra {label}{stride};");
            sb.AppendLine("ld.shared.f32 %f20, [%saddr];");
            sb.AppendLine($"ld.shared.f32 %f21, [%saddr+{stride * 4}];");
            sb.AppendLine("add.f32 %f20, %f20, %f21;");
            sb.AppendLine("st.shared.f32 [%saddr], %f20;");
            sb.AppendLine($"{label}{stride}:");
            sb.AppendLine("bar.sync 0;");
        }

        return sb.ToString();
    }

    /// <summary>Header of a one-block-per-group kernel: %g, %thr, %toff, %outer/%groups/%inner, %m = outer*inner.</summary>
    private static void GroupKernelStart(StringBuilder sb, string name, string[] pointers, string extraParams)
    {
        sb.AppendLine($".visible .entry {name}(");
        foreach (var p in pointers)
        {
            sb.AppendLine($"    .param .u64 p_{p},");
        }

        sb.AppendLine("    .param .u32 p_outer,");
        sb.AppendLine("    .param .u32 p_groups,");
        sb.Append("    .param .u32 p_inner");
        sb.AppendLine(extraParams);
        sb.AppendLine(")");
        sb.AppendLine("{");
        sb.AppendLine($"""
                .reg .pred %p<8>;
                .reg .f32 %f<32>;
                .reg .b32 %r<32>;
                .reg .b64 %rd<16>;
                .reg .u32 %g, %thr, %toff, %outer, %groups, %inner, %m, %j, %sbase, %saddr;
                .reg .u64 {string.Join(", ", pointers.Select(p => "%rd_" + p))};
                .shared .align 4 .f32 s1[{BlockSize}];
                .shared .align 4 .f32 s2[{BlockSize}];
                mov.u32 %g, %ctaid.x;
                mov.u32 %thr, %tid.x;
                shl.b32 %toff, %thr, 2;
                ld.param.u32 %outer, [p_outer];
                ld.param.u32 %groups, [p_groups];
                ld.param.u32 %inner, [p_inner];
                mul.lo.u32 %m, %outer, %inner;
            """);
        foreach (var p in pointers)
        {
            sb.AppendLine($"    ld.param.u64 %rd_{p}, [p_{p}];");
            sb.AppendLine($"    cvta.to.global.u64 %rd_{p}, %rd_{p};");
        }
    }

    private static void Normalization(StringBuilder sb)
    {
        // Mean, variance and 1/sqrt(var + eps) of each group; one block per group, two passes over its elements.
        string[] statsPointers = ["x", "mean", "var", "invstd"];
        GroupKernelStart(sb, "norm_stats_f32", statsPointers, ",\n    .param .f32 p_eps\n");
        sb.AppendLine("""
                mov.f32 %f1, 0f00000000;
                mov.u32 %j, %thr;
            PASS1:
                setp.ge.u32 %p1, %j, %m;
                @%p1 bra PASS1_END;
            """ + GroupElement("%j", "%r13") + $"""
                mul.wide.u32 %rd1, %r13, 4;
                add.u64 %rd1, %rd_x, %rd1;
                ld.global.f32 %f2, [%rd1];
                add.f32 %f1, %f1, %f2;
                add.u32 %j, %j, {BlockSize};
                bra PASS1;
            PASS1_END:
            """ + BlockReduce("s1", "%f1", "R1_") + """
                ld.shared.f32 %f3, [%sbase];
                cvt.rn.f32.u32 %f4, %m;
                div.rn.f32 %f5, %f3, %f4;
                bar.sync 0;
                mov.f32 %f1, 0f00000000;
                mov.u32 %j, %thr;
            PASS2:
                setp.ge.u32 %p1, %j, %m;
                @%p1 bra PASS2_END;
            """ + GroupElement("%j", "%r13") + $"""
                mul.wide.u32 %rd1, %r13, 4;
                add.u64 %rd1, %rd_x, %rd1;
                ld.global.f32 %f2, [%rd1];
                sub.f32 %f2, %f2, %f5;
                fma.rn.f32 %f1, %f2, %f2, %f1;
                add.u32 %j, %j, {BlockSize};
                bra PASS2;
            PASS2_END:
            """ + BlockReduce("s2", "%f1", "R2_") + """
                setp.ne.u32 %p3, %thr, 0;
                @%p3 bra DONE;
                ld.shared.f32 %f6, [%sbase];
                div.rn.f32 %f7, %f6, %f4;
                ld.param.f32 %f8, [p_eps];
                add.f32 %f9, %f7, %f8;
                sqrt.rn.f32 %f9, %f9;
                rcp.rn.f32 %f9, %f9;
                mul.wide.u32 %rd2, %g, 4;
                add.u64 %rd3, %rd_mean, %rd2;
                st.global.f32 [%rd3], %f5;
                add.u64 %rd3, %rd_var, %rd2;
                st.global.f32 [%rd3], %f7;
                add.u64 %rd3, %rd_invstd, %rd2;
                st.global.f32 [%rd3], %f9;
            DONE:
                ret;
            }
            """);

        // sumA[g] += Σ a; sumAB[g] += Σ a·b (when hasB). One block per group: no atomics needed.
        string[] reducePointers = ["a", "b", "suma", "sumab"];
        GroupKernelStart(sb, "group_reduce_f32", reducePointers, ",\n    .param .u32 p_hasb\n");
        sb.AppendLine("""
                ld.param.u32 %r20, [p_hasb];
                setp.ne.u32 %p4, %r20, 0;
                mov.f32 %f1, 0f00000000;
                mov.f32 %f10, 0f00000000;
                mov.u32 %j, %thr;
            LOOP:
                setp.ge.u32 %p1, %j, %m;
                @%p1 bra LOOP_END;
            """ + GroupElement("%j", "%r13") + $"""
                mul.wide.u32 %rd1, %r13, 4;
                add.u64 %rd2, %rd_a, %rd1;
                ld.global.f32 %f2, [%rd2];
                add.f32 %f1, %f1, %f2;
                @!%p4 bra NEXT;
                add.u64 %rd3, %rd_b, %rd1;
                ld.global.f32 %f3, [%rd3];
                fma.rn.f32 %f10, %f2, %f3, %f10;
            NEXT:
                add.u32 %j, %j, {BlockSize};
                bra LOOP;
            LOOP_END:
            """ + BlockReduce("s1", "%f1", "RA_") + """
                ld.shared.f32 %f11, [%sbase];
            """ + BlockReduce("s2", "%f10", "RB_") + """
                ld.shared.f32 %f12, [%sbase];
                setp.ne.u32 %p3, %thr, 0;
                @%p3 bra DONE;
                mul.wide.u32 %rd4, %g, 4;
                add.u64 %rd5, %rd_suma, %rd4;
                ld.global.f32 %f13, [%rd5];
                add.f32 %f13, %f13, %f11;
                st.global.f32 [%rd5], %f13;
                @!%p4 bra DONE;
                add.u64 %rd5, %rd_sumab, %rd4;
                ld.global.f32 %f13, [%rd5];
                add.f32 %f13, %f13, %f12;
                st.global.f32 [%rd5], %f13;
            DONE:
                ret;
            }
            """);

        const string Group = """
            div.u32 %r5, %i, %s_inner;
            rem.u32 %r5, %r5, %s_groups;
            mul.wide.u32 %rd1, %r5, 4;
            """;

        Elementwise(sb, "norm_apply_f32", ["x", "mean", "invstd", "y"], [("u32", "groups"), ("u32", "inner")],
            Group + """
            add.u64 %rd2, %b_mean, %rd1;
            add.u64 %rd3, %b_invstd, %rd1;
            ld.global.f32 %f1, [%a_x];
            ld.global.f32 %f2, [%rd2];
            ld.global.f32 %f3, [%rd3];
            sub.f32 %f4, %f1, %f2;
            mul.f32 %f4, %f4, %f3;
            st.global.f32 [%a_y], %f4;
            """);

        Elementwise(sb, "norm_bwd_f32", ["dxhat", "xhat", "s1", "s2", "invstd", "dx"], [("u32", "groups"), ("u32", "inner"), ("f32", "m")],
            Group + """
            add.u64 %rd2, %b_s1, %rd1;
            add.u64 %rd3, %b_s2, %rd1;
            add.u64 %rd4, %b_invstd, %rd1;
            ld.global.f32 %f1, [%a_dxhat];
            ld.global.f32 %f2, [%a_xhat];
            ld.global.f32 %f3, [%a_dx];
            ld.global.f32 %f4, [%rd2];
            ld.global.f32 %f5, [%rd3];
            ld.global.f32 %f6, [%rd4];
            mul.f32 %f7, %f1, %s_m;
            sub.f32 %f7, %f7, %f4;
            mul.f32 %f8, %f2, %f5;
            sub.f32 %f7, %f7, %f8;
            div.rn.f32 %f9, %f6, %s_m;
            fma.rn.f32 %f3, %f9, %f7, %f3;
            st.global.f32 [%a_dx], %f3;
            """);

        // flags: 1 = has scale, 2 = has shift, 4 = accumulate into y.
        Elementwise(sb, "group_scale_shift_f32", ["x", "scale", "shift", "y"], [("u32", "groups"), ("u32", "inner"), ("u32", "flags")],
            Group + $"""
            ld.global.f32 %f1, [%a_x];
            mov.f32 %f2, {One};
            mov.f32 %f3, {Zero};
            and.b32 %r6, %s_flags, 1;
            setp.eq.u32 %p1, %r6, 0;
            @%p1 bra NOSCALE;
            add.u64 %rd2, %b_scale, %rd1;
            ld.global.f32 %f2, [%rd2];
            NOSCALE:
            and.b32 %r6, %s_flags, 2;
            setp.eq.u32 %p1, %r6, 0;
            @%p1 bra NOSHIFT;
            add.u64 %rd3, %b_shift, %rd1;
            ld.global.f32 %f3, [%rd3];
            NOSHIFT:
            fma.rn.f32 %f4, %f1, %f2, %f3;
            and.b32 %r6, %s_flags, 4;
            setp.eq.u32 %p1, %r6, 0;
            @%p1 bra STORE;
            ld.global.f32 %f5, [%a_y];
            add.f32 %f4, %f4, %f5;
            STORE:
            st.global.f32 [%a_y], %f4;
            """);
    }

    // ------------------------------------------------------------------ embeddings

    /// <summary>Row index (clamped to maxindex) of element i: %r8 = table offset, element address in %rd3.</summary>
    private const string TableIndex = """
        div.u32 %r5, %i, %s_dim;
        rem.u32 %r6, %i, %s_dim;
        mul.wide.u32 %rd1, %r5, 4;
        add.u64 %rd2, %b_indices, %rd1;
        ld.global.f32 %f1, [%rd2];
        cvt.rzi.u32.f32 %r7, %f1;
        min.u32 %r7, %r7, %s_maxindex;
        mad.lo.u32 %r8, %r7, %s_dim, %r6;
        mul.wide.u32 %rd3, %r8, 4;
        """;

    private static void Embedding(StringBuilder sb)
    {
        Elementwise(sb, "gather_f32", ["table", "indices", "y"], [("u32", "dim"), ("u32", "maxindex")],
            TableIndex + """
            add.u64 %rd3, %b_table, %rd3;
            ld.global.f32 %f2, [%rd3];
            st.global.f32 [%a_y], %f2;
            """);

        // The same from a bfloat16 table packed two per word along each row (words per row = ⌈dim / 2⌉).
        Elementwise(sb, "gather_bf16_f32", ["table", "indices", "y"], [("u32", "dim"), ("u32", "maxindex"), ("u32", "words")], """
            div.u32 %r5, %i, %s_dim;
            rem.u32 %r6, %i, %s_dim;
            mul.wide.u32 %rd1, %r5, 4;
            add.u64 %rd2, %b_indices, %rd1;
            ld.global.f32 %f1, [%rd2];
            cvt.rzi.u32.f32 %r7, %f1;
            min.u32 %r7, %r7, %s_maxindex;
            shr.u32 %r9, %r6, 1;
            mad.lo.u32 %r8, %r7, %s_words, %r9;
            mul.wide.u32 %rd3, %r8, 4;
            add.u64 %rd3, %b_table, %rd3;
            ld.global.u32 %r10, [%rd3];
            and.b32 %r11, %r6, 1;
            setp.eq.u32 %p1, %r11, 0;
            shl.b32 %r12, %r10, 16;
            and.b32 %r13, %r10, 0xFFFF0000;
            selp.b32 %r12, %r12, %r13, %p1;
            mov.b32 %f2, %r12;
            st.global.f32 [%a_y], %f2;
            """);

        // One-hot rows: element i is 1 where its column equals its row's index (clamped as in gather_f32), else 0.
        Elementwise(sb, "one_hot_f32", ["indices", "y"], [("u32", "dim"), ("u32", "maxindex")], """
            div.u32 %r5, %i, %s_dim;
            rem.u32 %r6, %i, %s_dim;
            mul.wide.u32 %rd1, %r5, 4;
            add.u64 %rd2, %b_indices, %rd1;
            ld.global.f32 %f1, [%rd2];
            cvt.rzi.u32.f32 %r7, %f1;
            min.u32 %r7, %r7, %s_maxindex;
            setp.eq.u32 %p1, %r7, %r6;
            selp.f32 %f2, 0f3F800000, 0f00000000, %p1;
            st.global.f32 [%a_y], %f2;
            """);

        Elementwise(sb, "scatter_add_f32", ["dy", "indices", "dtable"], [("u32", "dim"), ("u32", "maxindex")],
            TableIndex + """
            add.u64 %rd3, %b_dtable, %rd3;
            ld.global.f32 %f2, [%a_dy];
            atom.global.add.f32 %f3, [%rd3], %f2;
            """);
    }

    // ------------------------------------------------------------------ convolution and pooling

    private static readonly (string, string)[] ConvParams =
    [
        ("u32", "C"), ("u32", "H"), ("u32", "W"), ("u32", "KW"), ("u32", "SH"), ("u32", "SW"), ("u32", "PH"), ("u32", "PW"),
        ("u32", "OW"), ("u32", "patch"), ("u32", "khkw"), ("u32", "ohow"),
    ];

    /// <summary>Decodes column-matrix element i; leaves %p1 = inside the image and %r17 = flat input index.</summary>
    private const string PatchIndex = """
        div.u32 %r5, %i, %s_patch;
        rem.u32 %r6, %i, %s_patch;
        div.u32 %r7, %r5, %s_ohow;
        rem.u32 %r8, %r5, %s_ohow;
        div.u32 %r9, %r8, %s_OW;
        rem.u32 %r10, %r8, %s_OW;
        div.u32 %r11, %r6, %s_khkw;
        rem.u32 %r12, %r6, %s_khkw;
        div.u32 %r13, %r12, %s_KW;
        rem.u32 %r14, %r12, %s_KW;
        mad.lo.s32 %r15, %r9, %s_SH, %r13;
        sub.s32 %r15, %r15, %s_PH;
        mad.lo.s32 %r16, %r10, %s_SW, %r14;
        sub.s32 %r16, %r16, %s_PW;
        setp.lt.u32 %p1, %r15, %s_H;
        setp.lt.and.u32 %p1, %r16, %s_W, %p1;
        mad.lo.u32 %r17, %r7, %s_C, %r11;
        mad.lo.u32 %r17, %r17, %s_H, %r15;
        mad.lo.u32 %r17, %r17, %s_W, %r16;
        mul.wide.u32 %rd1, %r17, 4;
        """;

    private static void Convolution(StringBuilder sb)
    {
        Elementwise(sb, "im2col_f32", ["x", "cols"], ConvParams,
            PatchIndex + $"""
            mov.f32 %f1, {Zero};
            @!%p1 bra STORE;
            add.u64 %rd2, %b_x, %rd1;
            ld.global.f32 %f1, [%rd2];
            STORE:
            st.global.f32 [%a_cols], %f1;
            """);

        Elementwise(sb, "col2im_f32", ["dcols", "dx"], ConvParams,
            PatchIndex + """
            @!%p1 bra SKIP;
            ld.global.f32 %f1, [%a_dcols];
            add.u64 %rd2, %b_dx, %rd1;
            atom.global.add.f32 %f2, [%rd2], %f1;
            SKIP:
            """);

        // One thread per output (n, c, oh, ow): scan the window, keep the first maximum.
        Elementwise(sb, "maxpool_f32", ["x", "y", "argmax"],
            [("u32", "H"), ("u32", "W"), ("u32", "KH"), ("u32", "KW"), ("u32", "SH"), ("u32", "SW"), ("u32", "PH"), ("u32", "PW"), ("u32", "OW"), ("u32", "ohow")],
            $"""
            div.u32 %r5, %i, %s_ohow;
            rem.u32 %r6, %i, %s_ohow;
            div.u32 %r7, %r6, %s_OW;
            rem.u32 %r8, %r6, %s_OW;
            mul.lo.u32 %r18, %s_H, %s_W;
            mul.lo.u32 %r18, %r18, %r5;
            mov.f32 %f1, {NegInf};
            mov.u32 %r20, %r18;
            mov.u32 %r9, 0;
            KH_LOOP:
            setp.ge.u32 %p1, %r9, %s_KH;
            @%p1 bra WINDOW_END;
            mad.lo.s32 %r10, %r7, %s_SH, %r9;
            sub.s32 %r10, %r10, %s_PH;
            setp.ge.u32 %p2, %r10, %s_H;
            @%p2 bra KH_NEXT;
            mov.u32 %r11, 0;
            KW_LOOP:
            setp.ge.u32 %p1, %r11, %s_KW;
            @%p1 bra KH_NEXT;
            mad.lo.s32 %r12, %r8, %s_SW, %r11;
            sub.s32 %r12, %r12, %s_PW;
            setp.ge.u32 %p2, %r12, %s_W;
            @%p2 bra KW_NEXT;
            mad.lo.u32 %r13, %r10, %s_W, %r12;
            add.u32 %r13, %r13, %r18;
            mul.wide.u32 %rd1, %r13, 4;
            add.u64 %rd1, %b_x, %rd1;
            ld.global.f32 %f2, [%rd1];
            setp.gt.f32 %p3, %f2, %f1;
            selp.f32 %f1, %f2, %f1, %p3;
            selp.u32 %r20, %r13, %r20, %p3;
            KW_NEXT:
            add.u32 %r11, %r11, 1;
            bra KW_LOOP;
            KH_NEXT:
            add.u32 %r9, %r9, 1;
            bra KH_LOOP;
            WINDOW_END:
            st.global.f32 [%a_y], %f1;
            st.global.u32 [%a_argmax], %r20;
            """);

        Elementwise(sb, "maxpool_bwd_f32", ["dy", "argmax", "dx"], [],
            """
            ld.global.u32 %r5, [%a_argmax];
            ld.global.f32 %f1, [%a_dy];
            mul.wide.u32 %rd1, %r5, 4;
            add.u64 %rd1, %b_dx, %rd1;
            atom.global.add.f32 %f2, [%rd1], %f1;
            """);
    }

    // ------------------------------------------------------------------ shape operations

    private static void ShapeKernels(StringBuilder sb)
    {
        // Output element i: decode coordinates right to left over 6 (right-aligned, 1-padded) dimensions.
        var scalars = new List<(string, string)>();
        for (int d = 0; d < MaxPermuteRank; d++)
        {
            scalars.Add(("u32", $"s{d}"));
        }

        for (int d = 0; d < MaxPermuteRank; d++)
        {
            scalars.Add(("u32", $"t{d}"));
        }

        scalars.Add(("u32", "accumulate"));
        var body = new StringBuilder("mov.u32 %r5, %i;\nmov.u32 %r6, 0;\n");
        for (int d = MaxPermuteRank - 1; d >= 0; d--)
        {
            body.AppendLine($"rem.u32 %r7, %r5, %s_s{d};");
            body.AppendLine($"div.u32 %r5, %r5, %s_s{d};");
            body.AppendLine($"mad.lo.u32 %r6, %r7, %s_t{d}, %r6;");
        }

        body.Append("""
            mul.wide.u32 %rd1, %r6, 4;
            add.u64 %rd1, %b_x, %rd1;
            ld.global.f32 %f1, [%rd1];
            setp.eq.u32 %p1, %s_accumulate, 0;
            @%p1 bra STORE;
            ld.global.f32 %f2, [%a_y];
            add.f32 %f1, %f1, %f2;
            STORE:
            st.global.f32 [%a_y], %f1;
            """);
        Elementwise(sb, "permute_f32", ["x", "y"], [.. scalars], body.ToString());

        Elementwise(sb, "copy2d_f32", ["src", "dst"],
            [("u32", "srcoff"), ("u32", "srcstride"), ("u32", "dstoff"), ("u32", "dststride"), ("u32", "cols"), ("u32", "accumulate")],
            """
            div.u32 %r5, %i, %s_cols;
            rem.u32 %r6, %i, %s_cols;
            mad.lo.u32 %r7, %r5, %s_srcstride, %r6;
            add.u32 %r7, %r7, %s_srcoff;
            mad.lo.u32 %r8, %r5, %s_dststride, %r6;
            add.u32 %r8, %r8, %s_dstoff;
            mul.wide.u32 %rd1, %r7, 4;
            add.u64 %rd1, %b_src, %rd1;
            mul.wide.u32 %rd2, %r8, 4;
            add.u64 %rd2, %b_dst, %rd2;
            ld.global.f32 %f1, [%rd1];
            setp.eq.u32 %p1, %s_accumulate, 0;
            @%p1 bra STORE;
            ld.global.f32 %f2, [%rd2];
            add.f32 %f1, %f1, %f2;
            STORE:
            st.global.f32 [%rd2], %f1;
            """);

        // y[o, j] (+)= scale * Σ_d x[o, d, j]; one thread per output element.
        Elementwise(sb, "sum_axis_f32", ["x", "y"], [("u32", "dim"), ("u32", "inner"), ("f32", "scale"), ("u32", "accumulate")],
            $"""
            div.u32 %r5, %i, %s_inner;
            rem.u32 %r6, %i, %s_inner;
            mul.lo.u32 %r7, %r5, %s_dim;
            mad.lo.u32 %r7, %r7, %s_inner, %r6;
            mul.wide.u32 %rd1, %r7, 4;
            add.u64 %rd1, %b_x, %rd1;
            mul.wide.u32 %rd2, %s_inner, 4;
            mov.f32 %f1, {Zero};
            mov.u32 %r8, 0;
            LOOP:
            setp.ge.u32 %p1, %r8, %s_dim;
            @%p1 bra LOOP_END;
            ld.global.f32 %f2, [%rd1];
            add.f32 %f1, %f1, %f2;
            add.u64 %rd1, %rd1, %rd2;
            add.u32 %r8, %r8, 1;
            bra LOOP;
            LOOP_END:
            mul.f32 %f1, %f1, %s_scale;
            setp.eq.u32 %p2, %s_accumulate, 0;
            @%p2 bra STORE;
            ld.global.f32 %f3, [%a_y];
            add.f32 %f1, %f1, %f3;
            STORE:
            st.global.f32 [%a_y], %f1;
            """);

        // dx[o, d, j] += scale * dy[o, j]; one thread per dx element.
        Elementwise(sb, "broadcast_axis_f32", ["dy", "dx"], [("u32", "diminner"), ("u32", "inner"), ("f32", "scale")],
            """
            div.u32 %r5, %i, %s_diminner;
            rem.u32 %r6, %i, %s_inner;
            mad.lo.u32 %r7, %r5, %s_inner, %r6;
            mul.wide.u32 %rd1, %r7, 4;
            add.u64 %rd1, %b_dy, %rd1;
            ld.global.f32 %f1, [%rd1];
            ld.global.f32 %f2, [%a_dx];
            fma.rn.f32 %f2, %f1, %s_scale, %f2;
            st.global.f32 [%a_dx], %f2;
            """);
    }
}
