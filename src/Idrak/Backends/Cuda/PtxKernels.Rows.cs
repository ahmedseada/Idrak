// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;

namespace Idrak.Backends.Cuda;

// PTX building blocks for kernels that give each row a block of threads (row reductions over thousands of columns,
// such as a softmax over a 150k-token vocabulary), and matrix products with few rows (token-by-token decoding).
internal static partial class PtxKernels
{
    public static readonly string[] RowNames = ["gemv_nn_f32", "gemv_nt_f32", "attention_decode_f32", "attention_decode_int8", "gemm128_f32", "gemm64_f32", "gemv_multi_f32", "attention_flash_f32", "attention_flash_int8", "attn_bwd_d_f32", "attn_bwd_kv_f32", "attn_bwd_q_f32", "attention_decode_bf16", "attention_flash_bf16", "gemm128_int8_f32", "gemm64_int8_f32", "gemm128_int4_f32", "gemm64_int4_f32", "gemm128_bf16_f32", "gemm64_bf16_f32", "adam8_f32", "transpose_f32", "sumsq_f32", "sum_cols_strided_f32", "multi_sumsq_f32", "multi_adamw_f32"];

    /// <summary>Threads of a <c>gemm128_f32</c> / <c>gemm64_f32</c> block.</summary>
    public const int GemmThreads = 256;

    /// <summary>Threads per row in <see cref="RowBlock"/> kernels (a multiple of 32, at most 1024).</summary>
    public const int RowThreads = 256;

    /// <summary>Threads of the sampler's block (one row: a whole vocabulary per block, so more loads in flight).</summary>
    private static int SamplerThreads => Shapes.SamplerThreads;

    /// <summary>Rows at most this many take the few-row matrix product kernels.</summary>
    public const int GemvRows = 8;

    /// <summary>
    /// A kernel with one block of <see cref="RowThreads"/> threads per row (row = block index, below the last
    /// parameter n). Parameters are laid out as in <see cref="Elementwise"/>, so launches pass the same arguments.
    /// The body sees %row (also in %i), %tx (thread index), %nt (threads per block), %lane, %warp, %nwarps, %b_x buffer bases,
    /// %a_x = %b_x + 4·row, %s_x scalars; %r/%f/%rd/%p registers are free except those the macros below document.
    /// </summary>
    private static void RowBlock(StringBuilder sb, string name, string[] pointers, (string Type, string Name)[] scalars, string body,
        int sharedFloats = 0)
    {
        var parameters = pointers.Select(p => $"    .param .u64 p_{p}")
            .Concat(scalars.Select(s => $"    .param .{s.Type} p_{s.Name}"))
            .Append("    .param .u32 p_n");

        sb.AppendLine($".visible .entry {name}(");
        sb.AppendLine(string.Join(",\n", parameters));
        sb.AppendLine(")");
        sb.AppendLine("{");
        sb.AppendLine("    .reg .pred %p<16>;");
        sb.AppendLine("    .reg .f32 %f<32>;");
        sb.AppendLine("    .reg .b32 %r<32>;");
        sb.AppendLine("    .reg .b64 %rd<16>;");
        sb.AppendLine("    .reg .u32 %i, %n, %row, %tx, %nt, %lane, %warp, %nwarps, %sb, %sbi, %spart;");
        sb.AppendLine("    .reg .u64 %off;");
        sb.AppendLine("    .reg .f32 %bf<4>;");
        sb.AppendLine("    .reg .b32 %br<6>;");
        sb.AppendLine("    .reg .pred %bp<4>;");
        sb.AppendLine($"    .shared .align 4 .b32 {name}_rv[{Shapes.MaxWarpsPerBlock}];");
        sb.AppendLine($"    .shared .align 4 .b32 {name}_ri[{Shapes.MaxWarpsPerBlock}];");
        if (sharedFloats > 0)
        {
            sb.AppendLine($"    .shared .align 4 .b32 {name}_part[{sharedFloats}];");
        }

        foreach (var p in pointers)
        {
            sb.AppendLine($"    .reg .u64 %a_{p}, %b_{p};");
        }

        foreach (var s in scalars)
        {
            sb.AppendLine($"    .reg .{s.Type} %s_{s.Name};");
        }

        sb.AppendLine($"""
                mov.u32 %row, %ctaid.x;
                ld.param.u32 %n, [p_n];
                setp.ge.u32 %p0, %row, %n;
                @%p0 bra DONE;
                mov.u32 %i, %row;
                mov.u32 %tx, %tid.x;
                mov.u32 %nt, %ntid.x;
                and.b32 %lane, %tx, 31;
                shr.u32 %warp, %tx, 5;
                add.u32 %nwarps, %nt, 31;
                shr.u32 %nwarps, %nwarps, 5;
                mov.u32 %sb, {name}_rv;
                mov.u32 %sbi, {name}_ri;
                mul.wide.u32 %off, %row, 4;
            """);
        if (sharedFloats > 0)
        {
            sb.AppendLine($"    mov.u32 %spart, {name}_part;");
        }

        foreach (var p in pointers)
        {
            sb.AppendLine($"    ld.param.u64 %b_{p}, [p_{p}];");
            sb.AppendLine($"    cvta.to.global.u64 %b_{p}, %b_{p};");
            sb.AppendLine($"    add.u64 %a_{p}, %b_{p}, %off;");
        }

        foreach (var s in scalars)
        {
            sb.AppendLine($"    ld.param.{s.Type} %s_{s.Name}, [p_{s.Name}];");
        }

        foreach (var line in body.Split('\n'))
        {
            sb.Append("    ").AppendLine(line.TrimEnd());
        }

        sb.AppendLine("DONE:");
        sb.AppendLine("    ret;");
        sb.AppendLine("}");
        sb.AppendLine();
    }

    /// <summary>
    /// Loop of this thread over columns tid, tid + nt, … below <paramref name="limit"/> of the row at
    /// <paramref name="rowPtr"/>: %r6 = column, %rd5 = its address, %f2 = its value. "bra {label}_NEXT" continues.
    /// </summary>
    private static string StridedLoop(string label, string rowPtr, string limit, string body) => $"""
        mov.u32 %r6, %tx;
        {label}:
        setp.ge.u32 %p1, %r6, {limit};
        @%p1 bra {label}_END;
        mul.wide.u32 %rd5, %r6, 4;
        add.u64 %rd5, %rd5, {rowPtr};
        ld.global.f32 %f2, [%rd5];
        {body}
        {label}_NEXT:
        add.u32 %r6, %r6, %nt;
        bra {label};
        {label}_END:
        """;

    /// <summary>
    /// Combines <paramref name="reg"/> over the block with add, max or min; every thread ends with the result. All
    /// threads of the block must reach it. Uses %bf*, %br*, %bp* and the block's reduction slots.
    /// </summary>
    private static string BlockReduce(string label, string reg, string op, string identity)
    {
        var s = new StringBuilder();
        foreach (int offset in new[] { 16, 8, 4, 2, 1 })
        {
            s.AppendLine($"shfl.sync.bfly.b32 %bf0, {reg}, {offset}, 31, 0xffffffff;");
            s.AppendLine($"{op}.f32 {reg}, {reg}, %bf0;");
        }

        s.Append($"""
            setp.eq.u32 %bp0, %lane, 0;
            shl.b32 %br0, %warp, 2;
            add.u32 %br0, %br0, %sb;
            @%bp0 st.shared.f32 [%br0], {reg};
            bar.sync 0;
            mov.f32 {reg}, {identity};
            mov.u32 %br1, 0;
            {label}:
            setp.ge.u32 %bp1, %br1, %nwarps;
            @%bp1 bra {label}_END;
            shl.b32 %br0, %br1, 2;
            add.u32 %br0, %br0, %sb;
            ld.shared.f32 %bf0, [%br0];
            {op}.f32 {reg}, {reg}, %bf0;
            add.u32 %br1, %br1, 1;
            bra {label};
            {label}_END:
            bar.sync 0;
            """);
        return s.ToString();
    }

    /// <summary>
    /// Arg-max over the block of (<paramref name="value"/>, <paramref name="index"/>) pairs: the larger value wins,
    /// on ties the smaller index (as unsigned, so -1 loses). Every thread ends with the winner.
    /// </summary>
    private static string BlockArgMax(string label, string value, string index)
    {
        const string Better = """
            setp.gt.f32 %bp0, %bf0, {0};
            setp.eq.f32 %bp1, %bf0, {0};
            setp.lt.u32 %bp2, %br2, {1};
            and.pred %bp1, %bp1, %bp2;
            or.pred %bp0, %bp0, %bp1;
            selp.f32 {0}, %bf0, {0}, %bp0;
            selp.b32 {1}, %br2, {1}, %bp0;
            """;
        var s = new StringBuilder();
        foreach (int offset in new[] { 16, 8, 4, 2, 1 })
        {
            s.AppendLine($"shfl.sync.bfly.b32 %bf0, {value}, {offset}, 31, 0xffffffff;");
            s.AppendLine($"shfl.sync.bfly.b32 %br2, {index}, {offset}, 31, 0xffffffff;");
            s.AppendLine(string.Format(Better, value, index));
        }

        s.Append($"""
            setp.eq.u32 %bp3, %lane, 0;
            shl.b32 %br0, %warp, 2;
            add.u32 %br3, %br0, %sb;
            @%bp3 st.shared.f32 [%br3], {value};
            add.u32 %br3, %br0, %sbi;
            @%bp3 st.shared.b32 [%br3], {index};
            bar.sync 0;
            ld.shared.f32 {value}, [%sb];
            ld.shared.b32 {index}, [%sbi];
            mov.u32 %br1, 1;
            {label}:
            setp.ge.u32 %bp3, %br1, %nwarps;
            @%bp3 bra {label}_END;
            shl.b32 %br0, %br1, 2;
            add.u32 %br3, %br0, %sb;
            ld.shared.f32 %bf0, [%br3];
            add.u32 %br3, %br0, %sbi;
            ld.shared.b32 %br2, [%br3];
            """ + "\n" + string.Format(Better, value, index) + "\n" + $"""
            add.u32 %br1, %br1, 1;
            bra {label};
            {label}_END:
            bar.sync 0;
            """);
        return s.ToString();
    }

    /// <summary>Largest head size <c>attention_decode_f32</c> handles (8 dimensions per lane).</summary>
    public const int DecodeMaxDim = 256;

    // 8-bit Adam (see Backend.AdamStep8Bit): one block of 256 threads per quantization block, one element per thread.
    // The two code tables go to shared memory; after the update the block's new scales are block maxima, and each
    // moment is encoded to its nearest code by binary search. Parameters: count = elements, n = blocks.
    private static void Adam8(StringBuilder sb)
    {
        string Encode(string label, string value, string table) => $"""
            mov.u32 %r20, 0;
            """ + string.Concat(new[] { 128, 64, 32, 16, 8, 4, 2, 1 }.Select(step => $"""

            add.u32 %r21, %r20, {step};
            shl.b32 %r22, %r21, 2;
            add.u32 %r22, %r22, {table};
            ld.shared.f32 %f20, [%r22];
            setp.le.f32 %p5, %f20, {value};
            selp.b32 %r20, %r21, %r20, %p5;
            """)) + $"""

            shl.b32 %r22, %r20, 2;
            add.u32 %r22, %r22, {table};
            ld.shared.f32 %f20, [%r22];
            setp.lt.u32 %p6, %r20, 255;
            @!%p6 bra {label}_DONE;
            ld.shared.f32 %f21, [%r22+4];
            sub.f32 %f21, %f21, {value};
            sub.f32 %f22, {value}, %f20;
            setp.lt.f32 %p6, %f21, %f22;
            @%p6 add.u32 %r20, %r20, 1;
            {label}_DONE:
            """;

        RowBlock(sb, "adam8_f32", ["p", "g", "m", "v", "absmax", "map"],
            [("f32", "lr"), ("f32", "b1"), ("f32", "b2"), ("f32", "c1"), ("f32", "c2"), ("f32", "eps"), ("u32", "count"), ("f32", "gscale"), ("f32", "decay")],
            $"""
            mul.wide.u32 %rd1, %tx, 4;
            add.u64 %rd2, %b_map, %rd1;
            ld.global.f32 %f1, [%rd2];
            ld.global.f32 %f2, [%rd2+1024];
            shl.b32 %r1, %tx, 2;
            add.u32 %r1, %r1, %spart;
            st.shared.f32 [%r1], %f1;
            st.shared.f32 [%r1+1024], %f2;
            bar.sync 0;
            add.u32 %r10, %spart, 1024;
            shl.b32 %r2, %row, 8;
            add.u32 %r2, %r2, %tx;
            setp.lt.u32 %p2, %r2, %s_count;
            cvt.u64.u32 %rd3, %r2;
            add.u64 %rd4, %b_m, %rd3;
            add.u64 %rd5, %b_v, %rd3;
            mul.wide.u32 %rd6, %r2, 4;
            add.u64 %rd7, %b_p, %rd6;
            add.u64 %rd8, %b_g, %rd6;
            ld.global.f32 %f3, [%a_absmax];
            mul.wide.u32 %rd9, %n, 4;
            add.u64 %rd10, %a_absmax, %rd9;
            ld.global.f32 %f4, [%rd10];
            mov.u32 %r3, 0;
            mov.u32 %r4, 0;
            mov.f32 %f5, 0f00000000;
            mov.f32 %f6, 0f00000000;
            @%p2 ld.global.u8 %r3, [%rd4];
            @%p2 ld.global.u8 %r4, [%rd5];
            @%p2 ld.global.f32 %f5, [%rd8];
            @%p2 ld.global.f32 %f6, [%rd7];
            mul.f32 %f5, %f5, %s_gscale;
            mul.f32 %f6, %f6, %s_decay;
            shl.b32 %r5, %r3, 2;
            add.u32 %r5, %r5, %spart;
            ld.shared.f32 %f7, [%r5];
            mul.f32 %f7, %f7, %f3;
            shl.b32 %r5, %r4, 2;
            add.u32 %r5, %r5, %r10;
            ld.shared.f32 %f8, [%r5];
            mul.f32 %f8, %f8, %f4;
            mul.f32 %f9, %f5, %s_c1;
            fma.rn.f32 %f7, %s_b1, %f7, %f9;
            mul.f32 %f9, %f5, %f5;
            mul.f32 %f9, %f9, %s_c2;
            fma.rn.f32 %f8, %s_b2, %f8, %f9;
            sqrt.rn.f32 %f10, %f8;
            add.f32 %f10, %f10, %s_eps;
            div.rn.f32 %f11, %f7, %f10;
            mul.f32 %f11, %f11, %s_lr;
            sub.f32 %f6, %f6, %f11;
            @%p2 st.global.f32 [%rd7], %f6;
            selp.f32 %f7, %f7, 0f00000000, %p2;
            selp.f32 %f8, %f8, 0f00000000, %p2;
            abs.f32 %f12, %f7;
            {BlockReduce("RM", "%f12", "max", Zero)}
            mov.f32 %f13, %f8;
            {BlockReduce("RV", "%f13", "max", Zero)}
            setp.gt.f32 %p3, %f12, 0f00000000;
            div.rn.f32 %f14, %f7, %f12;
            selp.f32 %f14, %f14, 0f00000000, %p3;
            setp.gt.f32 %p4, %f13, 0f00000000;
            div.rn.f32 %f15, %f8, %f13;
            selp.f32 %f15, %f15, 0f00000000, %p4;
            {Encode("EM", "%f14", "%spart")}
            @%p2 st.global.u8 [%rd4], %r20;
            {Encode("EV", "%f15", "%r10")}
            @%p2 st.global.u8 [%rd5], %r20;
            setp.eq.u32 %p7, %tx, 0;
            @%p7 st.global.f32 [%a_absmax], %f12;
            @%p7 st.global.f32 [%rd10], %f13;
            """, sharedFloats: 512);
    }

    // y[b][c, r] = x[b][r, c] for x [batch][rows, cols]: 32 × 32 tiles through shared memory (padded to 33, so both the
    // row-wise reads and the column-wise writes are conflict-free and coalesced). Block 32 × 8, grid (⌈cols / 32⌉,
    // ⌈rows / 32⌉, batch); batch strides in elements.
    private static void Transpose(StringBuilder sb)
    {
        var s = new StringBuilder();
        s.AppendLine("""
            .visible .entry transpose_f32(
                .param .u64 p_x, .param .u64 p_y, .param .u32 p_rows, .param .u32 p_cols, .param .u64 p_sx, .param .u64 p_sy
            )
            {
                .reg .pred %p<4>;
                .reg .b32 %r<24>;
                .reg .b64 %rd<12>;
                .reg .f32 %f<2>;
                .shared .align 4 .f32 transpose_f32_t[1056];
                ld.param.u64 %rd1, [p_x];
                ld.param.u64 %rd2, [p_y];
                cvta.to.global.u64 %rd1, %rd1;
                cvta.to.global.u64 %rd2, %rd2;
                ld.param.u32 %r1, [p_rows];
                ld.param.u32 %r2, [p_cols];
                ld.param.u64 %rd3, [p_sx];
                ld.param.u64 %rd4, [p_sy];
                mov.u32 %r3, %ctaid.z;
                cvt.u64.u32 %rd5, %r3;
                mul.lo.u64 %rd6, %rd5, %rd3;
                shl.b64 %rd6, %rd6, 2;
                add.u64 %rd1, %rd1, %rd6;
                mul.lo.u64 %rd6, %rd5, %rd4;
                shl.b64 %rd6, %rd6, 2;
                add.u64 %rd2, %rd2, %rd6;
                mov.u32 %r4, %tid.x;
                mov.u32 %r5, %tid.y;
                mov.u32 %r6, %ctaid.x;
                shl.b32 %r6, %r6, 5;
                mov.u32 %r7, %ctaid.y;
                shl.b32 %r7, %r7, 5;
                mov.u32 %r8, transpose_f32_t;
            """);
        for (int i = 0; i < 4; i++)
        {
            s.AppendLine($"""
                    add.u32 %r9, %r5, {8 * i};
                    add.u32 %r10, %r7, %r9;
                    add.u32 %r11, %r6, %r4;
                    setp.lt.u32 %p1, %r10, %r1;
                    setp.lt.and.u32 %p1, %r11, %r2, %p1;
                    mul.wide.u32 %rd7, %r10, %r2;
                    cvt.u64.u32 %rd8, %r11;
                    add.u64 %rd7, %rd7, %rd8;
                    shl.b64 %rd7, %rd7, 2;
                    add.u64 %rd7, %rd7, %rd1;
                    mov.f32 %f1, 0f00000000;
                    @%p1 ld.global.f32 %f1, [%rd7];
                    mul.lo.u32 %r12, %r9, 33;
                    add.u32 %r12, %r12, %r4;
                    shl.b32 %r12, %r12, 2;
                    add.u32 %r12, %r12, %r8;
                    st.shared.f32 [%r12], %f1;
                """);
        }

        s.AppendLine("bar.sync 0;");
        for (int i = 0; i < 4; i++)
        {
            s.AppendLine($"""
                    add.u32 %r9, %r5, {8 * i};
                    add.u32 %r10, %r6, %r9;
                    add.u32 %r11, %r7, %r4;
                    setp.lt.u32 %p1, %r10, %r2;
                    setp.lt.and.u32 %p1, %r11, %r1, %p1;
                    mul.lo.u32 %r12, %r4, 33;
                    add.u32 %r12, %r12, %r9;
                    shl.b32 %r12, %r12, 2;
                    add.u32 %r12, %r12, %r8;
                    ld.shared.f32 %f1, [%r12];
                    mul.wide.u32 %rd7, %r10, %r1;
                    cvt.u64.u32 %rd8, %r11;
                    add.u64 %rd7, %rd7, %rd8;
                    shl.b64 %rd7, %rd7, 2;
                    add.u64 %rd7, %rd7, %rd2;
                    @%p1 st.global.f32 [%rd7], %f1;
                """);
        }

        s.AppendLine("""
                ret;
            }
            """);
        sb.Append(s);
        sb.AppendLine();
    }

    // total[0] += Σ x²: blocks stride over x (n = blocks), reduce, and add their sums atomically.
    private static void SumSquares(StringBuilder sb) =>
        RowBlock(sb, "sumsq_f32", ["x", "total"], [("u32", "count")], $"""
            mov.f32 %f1, {Zero};
            mad.lo.u32 %r1, %row, %nt, %tx;
            mul.lo.u32 %r2, %n, %nt;
            SQ:
            setp.ge.u32 %p1, %r1, %s_count;
            @%p1 bra SQ_END;
            mul.wide.u32 %rd1, %r1, 4;
            add.u64 %rd1, %rd1, %b_x;
            ld.global.f32 %f2, [%rd1];
            fma.rn.f32 %f1, %f2, %f2, %f1;
            add.u32 %r1, %r1, %r2;
            bra SQ;
            SQ_END:
            {BlockReduce("RS", "%f1", "add", Zero)}
            setp.eq.u32 %p2, %tx, 0;
            @%p2 red.global.add.f32 [%b_total], %f1;
            """);

    // Many tensors in one launch (the optimizer over every adapter matrix): block b takes chunk b, 4096 elements of one
    // tensor, from p_chunks ([tensor index, first element] as u32 pairs). p_table holds per tensor 40 bytes: the parameter,
    // gradient, first- and second-moment pointers (u64) and the element count (u32).
    private const string ChunkStart = """
        mul.wide.u32 %rd1, %row, 8;
        add.u64 %rd1, %rd1, %b_chunks;
        ld.global.u32 %r10, [%rd1];
        ld.global.u32 %r11, [%rd1+4];
        mul.wide.u32 %rd2, %r10, 40;
        add.u64 %rd2, %rd2, %b_table;
        ld.global.u32 %r12, [%rd2+32];
        add.u32 %r13, %r11, 4096;
        min.u32 %r13, %r13, %r12;
        """;

    // total[0] += Σ g² over every tensor's gradient.
    private static void MultiSumSquares(StringBuilder sb) =>
        RowBlock(sb, "multi_sumsq_f32", ["chunks", "table", "total"], [], $"""
            {ChunkStart}
            ld.global.u64 %rd3, [%rd2+8];
            cvta.to.global.u64 %rd3, %rd3;
            mov.f32 %f1, {Zero};
            add.u32 %r1, %r11, %tx;
            MSQ:
            setp.ge.u32 %p1, %r1, %r13;
            @%p1 bra MSQ_END;
            mul.wide.u32 %rd4, %r1, 4;
            add.u64 %rd4, %rd4, %rd3;
            ld.global.f32 %f2, [%rd4];
            fma.rn.f32 %f1, %f2, %f2, %f1;
            add.u32 %r1, %r1, %nt;
            bra MSQ;
            MSQ_END:
            {BlockReduce("MS", "%f1", "add", Zero)}
            setp.eq.u32 %p2, %tx, 0;
            @%p2 red.global.add.f32 [%b_total], %f1;
            """);

    // AdamW over every tensor: g · factor[0] (the clipping factor), p · decay (decoupled weight decay), the moments and
    // the step as adam_f32 computes them; with p_zero ≠ 0 the gradient is zeroed for the next step.
    private static void MultiAdamW(StringBuilder sb) =>
        RowBlock(sb, "multi_adamw_f32", ["chunks", "table", "factor"],
            [("f32", "lr"), ("f32", "decay"), ("f32", "b1"), ("f32", "b2"), ("f32", "c1"), ("f32", "c2"), ("f32", "eps"), ("u32", "zero")], $"""
            {ChunkStart}
            ld.global.u64 %rd3, [%rd2];
            ld.global.u64 %rd4, [%rd2+8];
            ld.global.u64 %rd5, [%rd2+16];
            ld.global.u64 %rd6, [%rd2+24];
            cvta.to.global.u64 %rd3, %rd3;
            cvta.to.global.u64 %rd4, %rd4;
            cvta.to.global.u64 %rd5, %rd5;
            cvta.to.global.u64 %rd6, %rd6;
            ld.global.f32 %f10, [%b_factor];
            setp.ne.u32 %p3, %s_zero, 0;
            add.u32 %r1, %r11, %tx;
            MAD:
            setp.ge.u32 %p1, %r1, %r13;
            @%p1 bra MAD_END;
            mul.wide.u32 %rd7, %r1, 4;
            add.u64 %rd8, %rd3, %rd7;
            add.u64 %rd9, %rd4, %rd7;
            add.u64 %rd10, %rd5, %rd7;
            add.u64 %rd11, %rd6, %rd7;
            ld.global.f32 %f1, [%rd8];
            ld.global.f32 %f2, [%rd9];
            ld.global.f32 %f3, [%rd10];
            ld.global.f32 %f4, [%rd11];
            mul.f32 %f2, %f2, %f10;
            mul.f32 %f1, %f1, %s_decay;
            mul.f32 %f5, %f2, %s_c1;
            fma.rn.f32 %f3, %s_b1, %f3, %f5;
            mul.f32 %f6, %f2, %f2;
            mul.f32 %f6, %f6, %s_c2;
            fma.rn.f32 %f4, %s_b2, %f4, %f6;
            sqrt.rn.f32 %f7, %f4;
            add.f32 %f7, %f7, %s_eps;
            div.rn.f32 %f8, %f3, %f7;
            mul.f32 %f8, %f8, %s_lr;
            sub.f32 %f1, %f1, %f8;
            st.global.f32 [%rd10], %f3;
            st.global.f32 [%rd11], %f4;
            st.global.f32 [%rd8], %f1;
            @%p3 st.global.f32 [%rd9], {Zero};
            add.u32 %r1, %r1, %nt;
            bra MAD;
            MAD_END:
            """);

    // y[j] += Σ_r x[r·ld + j]: thread = column (x = blocks of BlockSize columns), grid y = chunks of rows, added atomically.
    private static void SumColumnsStrided(StringBuilder sb) => sb.AppendLine($$"""
        .visible .entry sum_cols_strided_f32(
            .param .u64 p_x, .param .u64 p_y, .param .u32 p_rows, .param .u32 p_cols, .param .u32 p_ld, .param .u32 p_chunk
        )
        {
            .reg .pred %p<4>;
            .reg .b32 %r<16>;
            .reg .b64 %rd<8>;
            .reg .f32 %f<4>;
            ld.param.u64 %rd1, [p_x];
            ld.param.u64 %rd2, [p_y];
            cvta.to.global.u64 %rd1, %rd1;
            cvta.to.global.u64 %rd2, %rd2;
            ld.param.u32 %r1, [p_rows];
            ld.param.u32 %r2, [p_cols];
            ld.param.u32 %r3, [p_ld];
            ld.param.u32 %r4, [p_chunk];
            mov.u32 %r5, %ctaid.x;
            mov.u32 %r6, %tid.x;
            mad.lo.u32 %r7, %r5, {{BlockSize}}, %r6;
            setp.ge.u32 %p1, %r7, %r2;
            @%p1 bra DONE;
            mov.u32 %r8, %ctaid.y;
            mul.lo.u32 %r9, %r8, %r4;
            add.u32 %r10, %r9, %r4;
            min.u32 %r10, %r10, %r1;
            mov.f32 %f1, 0f00000000;
        ROWS:
            setp.ge.u32 %p2, %r9, %r10;
            @%p2 bra ROWS_END;
            mul.wide.u32 %rd3, %r9, %r3;
            cvt.u64.u32 %rd4, %r7;
            add.u64 %rd3, %rd3, %rd4;
            shl.b64 %rd3, %rd3, 2;
            add.u64 %rd3, %rd3, %rd1;
            ld.global.f32 %f2, [%rd3];
            add.f32 %f1, %f1, %f2;
            add.u32 %r9, %r9, 1;
            bra ROWS;
        ROWS_END:
            mul.wide.u32 %rd5, %r7, 4;
            add.u64 %rd5, %rd5, %rd2;
            red.global.add.f32 [%rd5], %f1;
        DONE:
            ret;
        }
        """);

    private static void BuildRows(StringBuilder sb)
    {
        SumColumnsStrided(sb);
        SumSquares(sb);
        MultiSumSquares(sb);
        MultiAdamW(sb);
        Transpose(sb);
        Adam8(sb);
        GemvNN(sb);
        GemvNN(sb, multi: true);
        GemvNT(sb);
        AttentionDecode(sb);
        AttentionDecode(sb, int8: true);
        AttentionDecode(sb, bf16: true);
        AttentionFlash(sb);
        AttentionFlash(sb, int8: true);
        AttentionFlash(sb, bf16: true);
        AttentionBackward(sb);
        Gemm(sb, "gemm128_f32", 128, 8);
        Gemm(sb, "gemm64_f32", 64, 4);
        foreach (var (format, packed) in new[] { ("int8", 1), ("int4", 2), ("bf16", 3) })
        {
            Gemm(sb, $"gemm128_{format}_f32", 128, 8, packed);
            Gemm(sb, $"gemm64_{format}_f32", 64, 4, packed);
        }
    }

    /// <summary>Largest head size <c>attention_flash_f32</c> handles (4 dimensions per lane).</summary>
    public const int FlashMaxDim = 128;

    // The attention kernels' sliding window and soft-cap (see AttentionVariant): `window` 0 for none, else a query whose
    // causal limit is `end` (exclusive) starts at max(end - window, 0); `softcap` 0 for none, else each scaled score s
    // becomes cap · tanh(s / cap) before the softmax. Both are uniform per launch and predicated, so plain attention runs
    // the same instructions to the same results as before they were added.

    // %{start} = max(end - window, 0) with a window, else 0 (`pred` a scratch predicate).
    private static string WindowStartPtx(string start, string end, string window, string pred) => $"""
        mov.u32 {start}, 0;
        setp.gt.u32 {pred}, {end}, {window};
        setp.ne.and.u32 {pred}, {window}, 0, {pred};
        @{pred} sub.u32 {start}, {end}, {window};
        """;

    // value = cap · tanh(value / cap) where `capped` holds (cap > 0); tanh(x) = 1 - 2 / (2^(2x·log2 e) + 1), as tanh_f32
    // (±1 for large |x|). `temp` is a scratch float register.
    private static string SoftcapPtx(string value, string cap, string capped, string temp) => $"""
        @{capped} div.rn.f32 {temp}, {value}, {cap};
        @{capped} mul.f32 {temp}, {temp}, {F(2.8853900817779268f)};
        @{capped} ex2.approx.ftz.f32 {temp}, {temp};
        @{capped} add.f32 {temp}, {temp}, {One};
        @{capped} rcp.rn.f32 {temp}, {temp};
        @{capped} fma.rn.f32 {temp}, {temp}, {F(-2f)}, {One};
        @{capped} mul.f32 {value}, {temp}, {cap};
        """;

    // gradient ·= 1 - (score / cap)² where `capped` holds: the cap's slope, given the capped score.
    private static string SoftcapSlopePtx(string gradient, string score, string cap, string capped, string temp) => $"""
        @{capped} div.rn.f32 {temp}, {score}, {cap};
        @{capped} mul.f32 {temp}, {temp}, {temp};
        @{capped} sub.f32 {temp}, {One}, {temp};
        @{capped} mul.f32 {gradient}, {gradient}, {temp};
        """;

    /// <summary>Query rows per <c>attention_flash_f32</c> block, and key positions per tile.</summary>
    public const int FlashTile = 32;

    // Tiled attention (as in FlashAttention): o[h, i] = Σ_c softmax(scale · q[h, i] · k[h, c]) · v[h, c] over positions
    // c ≤ position[0] + (i % steps) (and < capacity), for q [heads, rowsPerHead, dim] and k, v [heads, capacity, dim];
    // with a window (p_window) from each row's window start (%lo per row; the block's tiles from its smallest), the
    // scores soft-capped with a cap (p_softcap).
    // A block of 4 warps takes 32 query rows (8 per warp) of one head and walks the keys in tiles of 32 positions held
    // in shared memory (keys transposed): lane p scores position p against the warp's 8 rows, the running maximum and
    // sum update per row (online softmax), and the lanes then split the head dimension to add Σ_p weight_p · v_p.
    // Every query row reads each key and value once per block instead of once per row. Optionally stores each row's
    // log-sum-exp (for a backward pass). Grid: x = ⌈rowsPerHead / 32⌉, y = heads.
    // int8: keys and values come from an int8 cache (see AttentionDecode) with per-row scales, dequantized as the tiles
    // are loaded into shared memory.
    // bf16: keys and values come from a bfloat16 cache (rows of `words` words, dimension d in half d % 2 of word d / 2).
    private static void AttentionFlash(StringBuilder sb, bool int8 = false, bool bf16 = false)
    {
        const int T = FlashTile, D = FlashMaxDim, Rows = 8;
        var s = new StringBuilder();
        s.AppendLine($$"""
            .visible .entry {{(int8 ? "attention_flash_int8" : bf16 ? "attention_flash_bf16" : "attention_flash_f32")}}(
                .param .u64 p_q, .param .u64 p_k, .param .u64 p_v, .param .u64 p_pos, .param .u64 p_o, .param .u64 p_lse,
                .param .u32 p_rph, .param .u32 p_steps, .param .u32 p_cap, .param .u32 p_dim, .param .f32 p_scale{{(int8 ? ",\n    .param .u64 p_ks, .param .u64 p_vs, .param .u32 p_words" : bf16 ? ",\n    .param .u32 p_words" : "")}},
                .param .u32 p_window, .param .f32 p_softcap
            )
            {
                .reg .pred %p<16>;
                .reg .b32 %lo<{{Rows}}>;
                .reg .b32 %window;
                .reg .f32 %softcap, %captmp;
                .reg .pred %capped, %windowed;
                .reg .f32 %o<{{Rows * 4}}>;
                .reg .f32 %m<{{Rows}}>;
                .reg .f32 %l<{{Rows}}>;
                .reg .f32 %s<{{Rows}}>;
                .reg .f32 %f<16>;
                .reg .b32 %r<48>;
                .reg .b64 %rd<24>;
                .reg .pred %dv<4>;
                .shared .align 16 .f32 fa_q[{{T * D}}];
                .shared .align 16 .f32 fa_k[{{T * D}}];
                .shared .align 16 .f32 fa_v[{{T * D}}];
                ld.param.u64 %rd1, [p_q];
                ld.param.u64 %rd2, [p_k];
                ld.param.u64 %rd3, [p_v];
                ld.param.u64 %rd4, [p_pos];
                ld.param.u64 %rd5, [p_o];
                ld.param.u64 %rd6, [p_lse];
                cvta.to.global.u64 %rd1, %rd1;
                cvta.to.global.u64 %rd2, %rd2;
                cvta.to.global.u64 %rd3, %rd3;
                cvta.to.global.u64 %rd4, %rd4;
                cvta.to.global.u64 %rd5, %rd5;
                setp.ne.u64 %p15, %rd6, 0;
                @%p15 cvta.to.global.u64 %rd6, %rd6;
                ld.param.u32 %r1, [p_rph];
                ld.param.u32 %r2, [p_steps];
                ld.param.u32 %r3, [p_cap];
                ld.param.u32 %r4, [p_dim];
                ld.param.f32 %f15, [p_scale];
                ld.param.u32 %window, [p_window];
                ld.param.f32 %softcap, [p_softcap];
                setp.gt.f32 %capped, %softcap, 0f00000000;
                ld.global.f32 %f1, [%rd4];
                cvt.rzi.u32.f32 %r5, %f1;
                mov.u32 %r6, %tid.x;
                and.b32 %r7, %r6, 31;
                shr.u32 %r8, %r6, 5;
                mov.u32 %r9, %ctaid.x;
                shl.b32 %r9, %r9, {{(int)Math.Log2(T)}};
                mov.u32 %r10, %ctaid.y;
                mul.lo.u32 %r11, %r10, %r1;
                mul.lo.u32 %r12, %r10, %r3;
                cvt.u64.u32 %rd7, %r11;
                cvt.u64.u32 %rd8, %r4;
                mul.lo.u64 %rd9, %rd7, %rd8;
                shl.b64 %rd9, %rd9, 2;
                add.u64 %rd10, %rd1, %rd9;
                add.u64 %rd11, %rd5, %rd9;
                cvt.u64.u32 %rd7, %r12;
                {{(int8 || bf16 ? "ld.param.u32 %r23, [p_words];\n    cvt.u64.u32 %rd20, %r23;\n    mul.lo.u64 %rd12, %rd7, %rd20;" : "mul.lo.u64 %rd12, %rd7, %rd8;")}}
                shl.b64 %rd12, %rd12, 2;
                add.u64 %rd13, %rd2, %rd12;
                add.u64 %rd14, %rd3, %rd12;
                mul.lo.u32 %r13, %r4, {{T}};
            """);
        if (int8)
        {
            // Scale rows of this head: ks/vs + head · capacity.
            s.AppendLine("""
                    ld.param.u64 %rd21, [p_ks];
                    ld.param.u64 %rd22, [p_vs];
                    cvta.to.global.u64 %rd21, %rd21;
                    cvta.to.global.u64 %rd22, %rd22;
                    mul.wide.u32 %rd23, %r12, 4;
                    add.u64 %rd21, %rd21, %rd23;
                    add.u64 %rd22, %rd22, %rd23;
                """);
        }
        // Query tile: fa_q[r, d] for rows row0 + r (zeros past the head's rows).
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
                mov.u32 %r20, fa_q;
                add.u32 %r19, %r19, %r20;
                st.shared.f32 [%r19], %f2;
                add.u32 %r14, %r14, 128;
                bra QL;
            QL_END:
            """);
        // Per-row limits (last position each row may see) and the block's last position.
        for (int i = 0; i < Rows; i++)
        {
            s.AppendLine($$"""
                    mad.lo.u32 %r21, %r8, {{Rows}}, {{i}};
                    add.u32 %r21, %r21, %r9;
                    rem.u32 %r22, %r21, %r2;
                    add.u32 %r{{24 + i}}, %r5, %r22;
                    sub.u32 %r21, %r3, 1;
                    min.u32 %r21, %r21, %r{{24 + i}};
                    add.u32 %r21, %r21, 1;
                    {{WindowStartPtx($"%lo{i}", "%r21", "%window", "%windowed")}}
                    mov.f32 %m{{i}}, 0fFF800000;
                    mov.f32 %l{{i}}, 0f00000000;
                """);
            for (int j = 0; j < 4; j++)
            {
                s.AppendLine($"    mov.f32 %o{i * 4 + j}, 0f00000000;");
            }
        }

        s.AppendLine($$"""
                add.u32 %r32, %r9, {{T - 1}};
                sub.u32 %r33, %r1, 1;
                min.u32 %r32, %r32, %r33;
                div.u32 %r34, %r9, %r2;
                div.u32 %r35, %r32, %r2;
                rem.u32 %r36, %r32, %r2;
                sub.u32 %r37, %r2, 1;
                setp.ne.u32 %p3, %r34, %r35;
                selp.b32 %r36, %r37, %r36, %p3;
                add.u32 %r36, %r36, %r5;
                sub.u32 %r33, %r3, 1;
                min.u32 %r36, %r36, %r33;
            """);
        for (int j = 0; j < 4; j++)
        {
            s.AppendLine($"    add.u32 %r38, %r7, {32 * j};");
            s.AppendLine($"    setp.lt.u32 %dv{j}, %r38, %r4;");
        }

        // The block's first tile: the window start of its smallest step (its first row's, or step 0 when it wraps).
        s.AppendLine($$"""
                rem.u32 %r37, %r9, %r2;
                selp.b32 %r37, 0, %r37, %p3;
                add.u32 %r37, %r37, %r5;
                sub.u32 %r38, %r3, 1;
                min.u32 %r37, %r37, %r38;
                add.u32 %r37, %r37, 1;
                {{WindowStartPtx("%r39", "%r37", "%window", "%windowed")}}
            TILE:
                setp.gt.u32 %p4, %r39, %r36;
                @%p4 bra TILE_END;
                bar.sync 0;
                mov.u32 %r14, %r6;
            KL:
                setp.ge.u32 %p1, %r14, %r13;
                @%p1 bra KL_END;
                div.u32 %r15, %r14, %r4;
                rem.u32 %r16, %r14, %r4;
                add.u32 %r17, %r39, %r15;
                setp.lt.u32 %p2, %r17, %r3;
                mov.f32 %f2, 0f00000000;
                mov.f32 %f3, 0f00000000;
            """);
        if (bf16)
        {
            // Word dim / 2 of the cached row, its half dim % 2 shifted to the top of a float.
            s.AppendLine("""
                    shr.u32 %r18, %r16, 1;
                    mad.lo.u32 %r18, %r17, %r23, %r18;
                    mul.wide.u32 %rd15, %r18, 4;
                    add.u64 %rd16, %rd15, %rd13;
                    add.u64 %rd17, %rd15, %rd14;
                    and.b32 %r22, %r16, 1;
                    xor.b32 %r22, %r22, 1;
                    shl.b32 %r22, %r22, 4;
                    @%p2 ld.global.u32 %r20, [%rd16];
                    @%p2 shl.b32 %r20, %r20, %r22;
                    @%p2 and.b32 %r20, %r20, 0xFFFF0000;
                    @%p2 mov.b32 %f2, %r20;
                    @%p2 ld.global.u32 %r20, [%rd17];
                    @%p2 shl.b32 %r20, %r20, %r22;
                    @%p2 and.b32 %r20, %r20, 0xFFFF0000;
                    @%p2 mov.b32 %f3, %r20;
                """);
        }
        else if (int8)
        {
            // Word dim / 4 of the cached row, byte dim % 4, times the row's scale.
            s.AppendLine("""
                    shr.u32 %r18, %r16, 2;
                    mad.lo.u32 %r18, %r17, %r23, %r18;
                    mul.wide.u32 %rd15, %r18, 4;
                    add.u64 %rd16, %rd15, %rd13;
                    add.u64 %rd17, %rd15, %rd14;
                    and.b32 %r22, %r16, 3;
                    shl.b32 %r22, %r22, 3;
                    @%p2 ld.global.u32 %r20, [%rd16];
                    @%p2 bfe.s32 %r20, %r20, %r22, 8;
                    @%p2 cvt.rn.f32.s32 %f2, %r20;
                    @%p2 ld.global.u32 %r20, [%rd17];
                    @%p2 bfe.s32 %r20, %r20, %r22, 8;
                    @%p2 cvt.rn.f32.s32 %f3, %r20;
                    mul.wide.u32 %rd15, %r17, 4;
                    add.u64 %rd16, %rd15, %rd21;
                    add.u64 %rd17, %rd15, %rd22;
                    @%p2 ld.global.f32 %f4, [%rd16];
                    @%p2 ld.global.f32 %f5, [%rd17];
                    @%p2 mul.f32 %f2, %f2, %f4;
                    @%p2 mul.f32 %f3, %f3, %f5;
                """);
        }
        else
        {
            s.AppendLine("""
                    mad.lo.u32 %r18, %r17, %r4, %r16;
                    mul.wide.u32 %rd15, %r18, 4;
                    add.u64 %rd16, %rd15, %rd13;
                    add.u64 %rd17, %rd15, %rd14;
                    @%p2 ld.global.f32 %f2, [%rd16];
                    @%p2 ld.global.f32 %f3, [%rd17];
                """);
        }
        s.AppendLine($$"""
                    mad.lo.u32 %r19, %r16, {{T}}, %r15;
                    shl.b32 %r19, %r19, 2;
                    mov.u32 %r20, fa_k;
                    add.u32 %r19, %r19, %r20;
                    st.shared.f32 [%r19], %f2;
                    mad.lo.u32 %r19, %r15, {{D}}, %r16;
                    shl.b32 %r19, %r19, 2;
                    mov.u32 %r20, fa_v;
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

        // Scores: lane = position; fa_k[d, lane], fa_q[warp·8 + i, d] (broadcast).
        s.AppendLine($$"""
                mov.u32 %r40, fa_k;
                shl.b32 %r41, %r7, 2;
                add.u32 %r40, %r40, %r41;
                mov.u32 %r42, fa_q;
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
                mov.u32 %r46, fa_v;
                shl.b32 %r47, %r7, 2;
                add.u32 %r46, %r46, %r47;
            """);
        for (int i = 0; i < Rows; i++)
        {
            // Mask, then the online softmax update for row i and its weighted values.
            s.AppendLine($$"""
                    setp.le.u32 %p6, %r45, %r{{24 + i}};
                    setp.lt.and.u32 %p6, %r45, %r3, %p6;
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

            s.AppendLine($"    mov.u32 %r47, %r46;");
            s.AppendLine("    mov.u32 %r20, 0;");
            s.AppendLine($"PV{i}:");
            s.AppendLine("    setp.ge.u32 %p8, %r20, 32;");
            s.AppendLine($"    @%p8 bra PV{i}_END;");
            s.AppendLine("    shfl.sync.idx.b32 %f13, %f11, %r20, 31, 0xffffffff;");
            for (int j = 0; j < 4; j++)
            {
                s.AppendLine($"    @%dv{j} ld.shared.f32 %f14, [%r47+{128 * j}];");
                s.AppendLine($"    @%dv{j} fma.rn.f32 %o{i * 4 + j}, %f13, %f14, %o{i * 4 + j};");
            }

            s.AppendLine($"    add.u32 %r47, %r47, {D * 4};");
            s.AppendLine("    add.u32 %r20, %r20, 1;");
            s.AppendLine($"    bra PV{i};");
            s.AppendLine($"PV{i}_END:");
        }

        s.AppendLine($$"""
                add.u32 %r39, %r39, {{T}};
                bra TILE;
            TILE_END:
            """);
        for (int i = 0; i < Rows; i++)
        {
            // o /= l; lse = m + ln l. Rows past the head's rows are not written.
            s.AppendLine($$"""
                    mad.lo.u32 %r21, %r8, {{Rows}}, {{i}};
                    add.u32 %r21, %r21, %r9;
                    setp.ge.u32 %p9, %r21, %r1;
                    @%p9 bra WRITTEN{{i}};
                    rcp.rn.f32 %f13, %l{{i}};
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
                    @!%p10 bra WRITTEN{{i}};
                    lg2.approx.ftz.f32 %f14, %l{{i}};
                    fma.rn.f32 %f14, %f14, 0f3F317218, %m{{i}};
                    add.u32 %r23, %r11, %r21;
                    mul.wide.u32 %rd19, %r23, 4;
                    add.u64 %rd19, %rd19, %rd6;
                    st.global.f32 [%rd19], %f14;
                WRITTEN{{i}}:
                """);
        }

        s.AppendLine("""
                ret;
            }
            """);
        sb.AppendLine(s.ToString());
    }

    // Backward of attention_flash_f32 (causal offset 0, keys and values [heads, capacity, dim]), recomputing the weights
    // from each row's log-sum-exp: P = exp(scale · q·k - lse), dS = P ∘ (dO·v - D) with D = Σ dO ∘ O per row, then
    // dV += Pᵀ dO, dK += scale · dSᵀ Q, dQ += scale · dS K. Three kernels, no atomics (deterministic):
    //   attn_bwd_d_f32   D per query row (one thread per row);
    //   attn_bwd_kv_f32  a block of 4 warps owns 16 keys (4 per warp) and walks the query rows in tiles of 32 held in
    //                    shared memory (rows padded to 129 floats, so lane-per-row and lane-per-dimension reads are
    //                    both conflict-free); grid x = ⌈capacity / 16⌉, y = heads;
    //   attn_bwd_q_f32   a block owns 32 query rows (8 per warp) and walks the keys in tiles of 32 held transposed in
    //                    shared memory (padded to 33); grid x = ⌈rowsPerHead / 32⌉, y = heads.
    private static void AttentionBackward(StringBuilder sb)
    {
        Elementwise(sb, "attn_bwd_d_f32", ["o", "dout", "dd"], [("u32", "dim")], $"""
            mul.lo.u32 %r5, %i, %s_dim;
            mul.wide.u32 %rd1, %r5, 4;
            add.u64 %rd2, %b_o, %rd1;
            add.u64 %rd3, %b_dout, %rd1;
            mov.f32 %f1, {Zero};
            mov.u32 %r6, 0;
            DL:
            setp.ge.u32 %p1, %r6, %s_dim;
            @%p1 bra DL_END;
            ld.global.f32 %f2, [%rd2];
            ld.global.f32 %f3, [%rd3];
            fma.rn.f32 %f1, %f2, %f3, %f1;
            add.u64 %rd2, %rd2, 4;
            add.u64 %rd3, %rd3, 4;
            add.u32 %r6, %r6, 1;
            bra DL;
            DL_END:
            st.global.f32 [%a_dd], %f1;
            """);
        AttentionBackwardKeys(sb);
        AttentionBackwardQueries(sb);
    }

    // Common prologue: pointers of this head (q, dout, dq [heads, rph, dim]; k, v, dk, dv [heads, cap, dim]; lse, dd
    // [heads, rph]), %r1 rph, %r2 steps, %r3 cap, %r4 dim, %f31 scale, %r6 tid, %r7 lane, %r8 warp, %r10 head; the
    // window in %r9 and the soft-cap in %f30, %p15 when there is a cap (see WindowStartPtx). With a window, row t sees key
    // p when t - p < window; with a cap, P = exp(s - lse) for the capped score s and dS takes the cap's slope.
    private const string BackwardParameters = """
            .param .u64 p_q, .param .u64 p_k, .param .u64 p_v, .param .u64 p_dout, .param .u64 p_lse, .param .u64 p_dd,
            .param .u64 p_dq, .param .u64 p_dk, .param .u64 p_dv,
            .param .u32 p_rph, .param .u32 p_steps, .param .u32 p_cap, .param .u32 p_dim, .param .f32 p_scale,
            .param .u32 p_window, .param .f32 p_softcap
        """;

    private static string BackwardPrologue => """
            ld.param.u32 %r1, [p_rph];
            ld.param.u32 %r2, [p_steps];
            ld.param.u32 %r3, [p_cap];
            ld.param.u32 %r4, [p_dim];
            ld.param.f32 %f31, [p_scale];
            ld.param.u32 %r9, [p_window];
            ld.param.f32 %f30, [p_softcap];
            setp.gt.f32 %p15, %f30, 0f00000000;
            mov.u32 %r6, %tid.x;
            and.b32 %r7, %r6, 31;
            shr.u32 %r8, %r6, 5;
            mov.u32 %r10, %ctaid.y;
            cvt.u64.u32 %rd30, %r4;
            mul.lo.u32 %r11, %r10, %r1;
            cvt.u64.u32 %rd29, %r11;
            mul.lo.u64 %rd28, %rd29, %rd30;
            shl.b64 %rd28, %rd28, 2;
            mul.lo.u32 %r12, %r10, %r3;
            cvt.u64.u32 %rd29, %r12;
            mul.lo.u64 %rd27, %rd29, %rd30;
            shl.b64 %rd27, %rd27, 2;
            ld.param.u64 %rd1, [p_q];
            cvta.to.global.u64 %rd1, %rd1;
            add.u64 %rd1, %rd1, %rd28;
            ld.param.u64 %rd4, [p_dout];
            cvta.to.global.u64 %rd4, %rd4;
            add.u64 %rd4, %rd4, %rd28;
            ld.param.u64 %rd7, [p_dq];
            cvta.to.global.u64 %rd7, %rd7;
            add.u64 %rd7, %rd7, %rd28;
            ld.param.u64 %rd2, [p_k];
            cvta.to.global.u64 %rd2, %rd2;
            add.u64 %rd2, %rd2, %rd27;
            ld.param.u64 %rd3, [p_v];
            cvta.to.global.u64 %rd3, %rd3;
            add.u64 %rd3, %rd3, %rd27;
            ld.param.u64 %rd8, [p_dk];
            cvta.to.global.u64 %rd8, %rd8;
            add.u64 %rd8, %rd8, %rd27;
            ld.param.u64 %rd9, [p_dv];
            cvta.to.global.u64 %rd9, %rd9;
            add.u64 %rd9, %rd9, %rd27;
            mul.wide.u32 %rd26, %r11, 4;
            ld.param.u64 %rd5, [p_lse];
            cvta.to.global.u64 %rd5, %rd5;
            add.u64 %rd5, %rd5, %rd26;
            ld.param.u64 %rd6, [p_dd];
            cvta.to.global.u64 %rd6, %rd6;
            add.u64 %rd6, %rd6, %rd26;
        """;

    private static void AttentionBackwardKeys(StringBuilder sb)
    {
        const int Keys = 4, Stride = 129, QT = 32;
        var s = new StringBuilder();
        s.AppendLine($$"""
            .visible .entry attn_bwd_kv_f32(
            {{BackwardParameters}}
            )
            {
                .reg .pred %p<16>;
                .reg .pred %dv<4>;
                .reg .f32 %gk<{{Keys * 4}}>;
                .reg .f32 %gv<{{Keys * 4}}>;
                .reg .f32 %sp<{{Keys}}>;
                .reg .f32 %sd<{{Keys}}>;
                .reg .f32 %f<32>;
                .reg .b32 %r<48>;
                .reg .b64 %rd<32>;
                .shared .align 4 .f32 kv_q[{{QT * Stride}}];
                .shared .align 4 .f32 kv_do[{{QT * Stride}}];
            {{BackwardPrologue}}
                mov.u32 %r13, %ctaid.x;
                shl.b32 %r13, %r13, 4;
                shl.b32 %r14, %r8, 2;
                add.u32 %r14, %r14, %r13;
                mul.lo.u32 %r15, %r4, {{QT}};
                mov.u32 %r16, kv_q;
                mov.u32 %r17, kv_do;
            """);
        for (int j = 0; j < 4; j++)
        {
            s.AppendLine($"    add.u32 %r18, %r7, {32 * j};");
            s.AppendLine($"    setp.lt.u32 %dv{j}, %r18, %r4;");
        }

        for (int i = 0; i < Keys * 4; i++)
        {
            s.AppendLine($"    mov.f32 %gk{i}, 0f00000000;");
            s.AppendLine($"    mov.f32 %gv{i}, 0f00000000;");
        }

        // Query tiles; a tile is skipped when none of its rows reaches this block's first key.
        s.AppendLine($$"""
                mov.u32 %r19, 0;
            QT:
                setp.ge.u32 %p1, %r19, %r1;
                @%p1 bra QT_END;
                add.u32 %r20, %r19, {{QT - 1}};
                sub.u32 %r21, %r1, 1;
                min.u32 %r20, %r20, %r21;
                div.u32 %r21, %r19, %r2;
                div.u32 %r22, %r20, %r2;
                rem.u32 %r23, %r20, %r2;
                sub.u32 %r24, %r2, 1;
                setp.ne.u32 %p2, %r21, %r22;
                selp.b32 %r23, %r24, %r23, %p2;
                setp.lt.u32 %p2, %r23, %r13;
                @%p2 bra QT_NEXT;
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
                add.u32 %r32, %r19, %r7;
                setp.lt.u32 %p5, %r32, %r1;
                rem.u32 %r33, %r32, %r2;
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
            // Key p = %r14 + kk: s = q_lane · k_p, dp = dO_lane · v_p (k, v rows broadcast from global memory).
            s.AppendLine($$"""
                    add.u32 %r37, %r14, {{kk}};
                    setp.lt.u32 %p6, %r37, %r3;
                    min.u32 %r38, %r37, %r3;
                    sub.u32 %r39, %r3, 1;
                    min.u32 %r38, %r38, %r39;
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
                    setp.le.u32 %p8, %r37, %r33;
                    and.pred %p8, %p8, %p5;
                    and.pred %p8, %p8, %p6;
                    sub.u32 %r5, %r33, %r37;
                    setp.lt.u32 %p14, %r5, %r9;
                    setp.eq.or.u32 %p14, %r9, 0, %p14;
                    and.pred %p8, %p8, %p14;
                    mul.f32 %f11, %f5, %f31;
                    {{SoftcapPtx("%f11", "%f30", "%p15", "%f26")}}
                    mov.f32 %f25, %f11;
                    sub.f32 %f11, %f11, %f3;
                    mul.f32 %f11, %f11, 0f3FB8AA3B;
                    ex2.approx.ftz.f32 %f11, %f11;
                    selp.f32 %sp{{kk}}, %f11, 0f00000000, %p8;
                    sub.f32 %f12, %f6, %f4;
                    mul.f32 %sd{{kk}}, %sp{{kk}}, %f12;
                    {{SoftcapSlopePtx($"%sd{kk}", "%f25", "%f30", "%p15", "%f26")}}
                """);
        }

        // dV_p += Σ_r P[r, p] dO[r], dK_p += Σ_r dS[r, p] Q[r] with the lanes splitting the dimension.
        s.AppendLine($$"""
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
            s.AppendLine($"    @%dv{j} ld.shared.f32 %f{13 + j}, [%r46+{128 * j}];");
            s.AppendLine($"    @%dv{j} ld.shared.f32 %f{17 + j}, [%r47+{128 * j}];");
        }

        for (int kk = 0; kk < Keys; kk++)
        {
            s.AppendLine($"    shfl.sync.idx.b32 %f21, %sp{kk}, %r44, 31, 0xffffffff;");
            s.AppendLine($"    shfl.sync.idx.b32 %f22, %sd{kk}, %r44, 31, 0xffffffff;");
            for (int j = 0; j < 4; j++)
            {
                s.AppendLine($"    fma.rn.f32 %gv{kk * 4 + j}, %f21, %f{17 + j}, %gv{kk * 4 + j};");
                s.AppendLine($"    fma.rn.f32 %gk{kk * 4 + j}, %f22, %f{13 + j}, %gk{kk * 4 + j};");
            }
        }

        s.AppendLine($$"""
                add.u32 %r44, %r44, 1;
                bra RR;
            RR_END:
            QT_NEXT:
                add.u32 %r19, %r19, {{QT}};
                bra QT;
            QT_END:
            """);
        for (int kk = 0; kk < Keys; kk++)
        {
            s.AppendLine($$"""
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
                s.AppendLine($$"""
                        @%dv{{j}} ld.global.f32 %f23, [%rd19+{{128 * j}}];
                        @%dv{{j}} fma.rn.f32 %f23, %gk{{kk * 4 + j}}, %f31, %f23;
                        @%dv{{j}} st.global.f32 [%rd19+{{128 * j}}], %f23;
                        @%dv{{j}} ld.global.f32 %f24, [%rd20+{{128 * j}}];
                        @%dv{{j}} add.f32 %f24, %f24, %gv{{kk * 4 + j}};
                        @%dv{{j}} st.global.f32 [%rd20+{{128 * j}}], %f24;
                    """);
            }

            s.AppendLine($"KW{kk}:");
        }

        s.AppendLine("""
                ret;
            }
            """);
        sb.AppendLine(s.ToString());
    }

    private static void AttentionBackwardQueries(StringBuilder sb)
    {
        const int Rows = 8, KT = 32, Stride = 33, D = FlashMaxDim;
        var s = new StringBuilder();
        s.AppendLine($$"""
            .visible .entry attn_bwd_q_f32(
            {{BackwardParameters}}
            )
            {
                .reg .pred %p<16>;
                .reg .pred %dv<4>;
                .reg .f32 %gq<{{Rows * 4}}>;
                .reg .f32 %f<32>;
                .reg .b32 %r<48>;
                .reg .b64 %rd<32>;
                .shared .align 4 .f32 bq_k[{{D * Stride}}];
                .shared .align 4 .f32 bq_v[{{D * Stride}}];
            {{BackwardPrologue}}
                mov.u32 %r13, %ctaid.x;
                shl.b32 %r13, %r13, 5;
                mul.lo.u32 %r15, %r4, {{KT}};
                mov.u32 %r16, bq_k;
                mov.u32 %r17, bq_v;
                add.u32 %r20, %r13, {{KT - 1}};
                sub.u32 %r21, %r1, 1;
                min.u32 %r20, %r20, %r21;
                div.u32 %r21, %r13, %r2;
                div.u32 %r22, %r20, %r2;
                rem.u32 %r23, %r20, %r2;
                sub.u32 %r24, %r2, 1;
                setp.ne.u32 %p2, %r21, %r22;
                selp.b32 %r23, %r24, %r23, %p2;
                sub.u32 %r24, %r3, 1;
                min.u32 %r23, %r23, %r24;
            """);
        for (int j = 0; j < 4; j++)
        {
            s.AppendLine($"    add.u32 %r18, %r7, {32 * j};");
            s.AppendLine($"    setp.lt.u32 %dv{j}, %r18, %r4;");
        }

        for (int i = 0; i < Rows * 4; i++)
        {
            s.AppendLine($"    mov.f32 %gq{i}, 0f00000000;");
        }

        s.AppendLine($$"""
                mov.u32 %r19, 0;
            KT:
                setp.gt.u32 %p1, %r19, %r23;
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
            // Row r = block row + warp·8 + i: s = q_r · k_lane, dp = dO_r · v_lane (q, dO rows broadcast from global).
            s.AppendLine($$"""
                    mad.lo.u32 %r35, %r8, {{Rows}}, {{i}};
                    add.u32 %r35, %r35, %r13;
                    setp.lt.u32 %p6, %r35, %r1;
                    min.u32 %r36, %r35, %r21;
                    sub.u32 %r36, %r1, 1;
                    min.u32 %r36, %r35, %r36;
                    rem.u32 %r37, %r36, %r2;
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
                    setp.le.u32 %p8, %r32, %r37;
                    and.pred %p8, %p8, %p5;
                    and.pred %p8, %p8, %p6;
                    sub.u32 %r5, %r37, %r32;
                    setp.lt.u32 %p14, %r5, %r9;
                    setp.eq.or.u32 %p14, %r9, 0, %p14;
                    and.pred %p8, %p8, %p14;
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
                    mov.u32 %r45, %r34;
                    add.u32 %r45, %r45, %r16;
                    mul.lo.u32 %r46, %r7, {{Stride * 4}};
                    add.u32 %r46, %r46, %r16;
                PQ{{i}}:
                    setp.ge.u32 %p9, %r44, {{KT}};
                    @%p9 bra PQ{{i}}_END;
                    shfl.sync.idx.b32 %f13, %f12, %r44, 31, 0xffffffff;
                """);
            for (int j = 0; j < 4; j++)
            {
                s.AppendLine($"    @%dv{j} ld.shared.f32 %f14, [%r46+{32 * Stride * 4 * j}];");
                s.AppendLine($"    @%dv{j} fma.rn.f32 %gq{i * 4 + j}, %f13, %f14, %gq{i * 4 + j};");
            }

            s.AppendLine($$"""
                    add.u32 %r46, %r46, 4;
                    add.u32 %r44, %r44, 1;
                    bra PQ{{i}};
                PQ{{i}}_END:
                """);
        }

        s.AppendLine($$"""
                add.u32 %r19, %r19, {{KT}};
                bra KT;
            KT_END:
            """);
        for (int i = 0; i < Rows; i++)
        {
            s.AppendLine($$"""
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
                s.AppendLine($$"""
                        @%dv{{j}} ld.global.f32 %f23, [%rd19+{{128 * j}}];
                        @%dv{{j}} fma.rn.f32 %f23, %gq{{i * 4 + j}}, %f31, %f23;
                        @%dv{{j}} st.global.f32 [%rd19+{{128 * j}}], %f23;
                    """);
            }

            s.AppendLine($"QW{i}:");
        }

        s.AppendLine("""
                ret;
            }
            """);
        sb.AppendLine(s.ToString());
    }

    // C[b] = op(A[b]) · op(B[b]) (+ beta · C) with register blocking: a block computes a tile × tile patch of C with
    // 256 threads, each an (per × per) patch, walking k eight at a time through shared memory (A stored k-major so a
    // thread's rows are contiguous; B likewise for its columns). Every output adds its k terms in ascending order with
    // fused multiply-adds, as the 16 × 16 kernel does, so results do not depend on which kernel ran.
    // Parameters as matmul_f32: a, b, c, m, n, k, transA, transB, beta, batch strides. Grid: x = ⌈n / tile⌉,
    // y = ⌈m / tile⌉, z = batch.
    //
    // packed 1 / 2 / 3: B is int8 (per-column scales), 4-bit (a scale per 32 rows and column) or bfloat16 weights [k, n]
    // packed as for Int8MatMul / Int4MatMul / BFloat16MatMul, expanded as the tile is loaded (prompts through quantized
    // layers without a float copy of the weights); not transposed. p_scales follows p_sc.
    private static void Gemm(StringBuilder sb, string name, int tile, int per, int packed = 0)
    {
        const int K = 8;
        int log = (int)Math.Log2(tile), loads = tile * K / GemmThreads, threadsPerRow = tile / per;
        var s = new StringBuilder();
        s.AppendLine($$"""
            .visible .entry {{name}}(
                .param .u64 p_a, .param .u64 p_b, .param .u64 p_c,
                .param .u32 p_m, .param .u32 p_n, .param .u32 p_k, .param .u32 p_ta, .param .u32 p_tb, .param .f32 p_beta,
                .param .u64 p_sa, .param .u64 p_sb, .param .u64 p_sc{{(packed > 0 ? ", .param .u64 p_scales" : "")}}
            )
            {
                .reg .pred %p<16>;
                .reg .f32 %acc<{{per * per}}>;
                .reg .f32 %fa<{{per}}>;
                .reg .f32 %fb<{{per}}>;
                .reg .f32 %f<8>;
                .reg .f32 %beta;
                .reg .b32 %r<48>;
                .reg .b64 %rd<32>;
                .shared .align 16 .f32 {{name}}_as[{{K * tile}}];
                .shared .align 16 .f32 {{name}}_bs[{{K * tile}}];
                ld.param.u64 %rd1, [p_a];
                ld.param.u64 %rd2, [p_b];
                ld.param.u64 %rd3, [p_c];
                cvta.to.global.u64 %rd1, %rd1;
                cvta.to.global.u64 %rd2, %rd2;
                cvta.to.global.u64 %rd3, %rd3;
                ld.param.u32 %r1, [p_m];
                ld.param.u32 %r2, [p_n];
                ld.param.u32 %r3, [p_k];
                ld.param.u32 %r4, [p_ta];
                ld.param.u32 %r5, [p_tb];
                ld.param.f32 %beta, [p_beta];
                ld.param.u64 %rd4, [p_sa];
                ld.param.u64 %rd5, [p_sb];
                ld.param.u64 %rd6, [p_sc];
                {{(packed > 0 ? "ld.param.u64 %rd29, [p_scales];\n    cvta.to.global.u64 %rd29, %rd29;" : "")}}
                {{(packed switch { 1 => "add.u32 %r41, %r2, 3;\n    shr.u32 %r41, %r41, 2;", 2 => "add.u32 %r41, %r2, 7;\n    shr.u32 %r41, %r41, 3;", 3 => "add.u32 %r41, %r2, 1;\n    shr.u32 %r41, %r41, 1;", _ => "" })}}
                mov.u32 %r40, %ctaid.z;
                cvt.u64.u32 %rd7, %r40;
                mul.lo.u64 %rd8, %rd7, %rd4;
                shl.b64 %rd8, %rd8, 2;
                add.u64 %rd1, %rd1, %rd8;
                mul.lo.u64 %rd8, %rd7, %rd5;
                shl.b64 %rd8, %rd8, 2;
                add.u64 %rd2, %rd2, %rd8;
                mul.lo.u64 %rd8, %rd7, %rd6;
                shl.b64 %rd8, %rd8, 2;
                add.u64 %rd3, %rd3, %rd8;
                setp.ne.u32 %p1, %r4, 0;
                setp.ne.u32 %p2, %r5, 0;
                setp.ne.f32 %p7, %beta, 0f00000000;
                mov.u32 %r6, %tid.x;
                and.b32 %r7, %r6, {{threadsPerRow - 1}};
                shr.u32 %r8, %r6, {{(int)Math.Log2(threadsPerRow)}};
                mov.u32 %r9, %ctaid.y;
                shl.b32 %r9, %r9, {{log}};
                mov.u32 %r10, %ctaid.x;
                shl.b32 %r10, %r10, {{log}};
                mov.u32 %r12, {{name}}_as;
                mov.u32 %r13, {{name}}_bs;
                shl.b32 %r31, %r8, {{(int)Math.Log2(per * 4)}};
                add.u32 %r31, %r31, %r12;
                shl.b32 %r32, %r7, {{(int)Math.Log2(per * 4)}};
                add.u32 %r32, %r32, %r13;
            """);
        for (int i = 0; i < per * per; i++)
        {
            s.AppendLine($"    mov.f32 %acc{i}, 0f00000000;");
        }

        s.AppendLine("""
                mov.u32 %r11, 0;
            KLOOP:
                setp.ge.u32 %p10, %r11, %r3;
                @%p10 bra KEND;
            """);
        for (int r = 0; r < loads; r++)
        {
            // A: element e of the tile; not transposed: (i = e / 8, kk = e % 8) with a[i, k] at i·K + k; transposed:
            // (kk = e / tile, i = e % tile) with a stored [K, m]. Stored in shared memory as as[kk, i].
            s.AppendLine($$"""
                    add.u32 %r20, %r6, {{GemmThreads * r}};
                    shr.u32 %r21, %r20, 3;
                    and.b32 %r22, %r20, 7;
                    shr.u32 %r23, %r20, {{log}};
                    and.b32 %r24, %r20, {{tile - 1}};
                    selp.b32 %r25, %r24, %r21, %p1;
                    selp.b32 %r26, %r23, %r22, %p1;
                    add.u32 %r27, %r9, %r25;
                    add.u32 %r28, %r11, %r26;
                    setp.lt.u32 %p3, %r27, %r1;
                    setp.lt.u32 %p4, %r28, %r3;
                    and.pred %p3, %p3, %p4;
                    mul.wide.u32 %rd10, %r27, %r3;
                    cvt.u64.u32 %rd11, %r28;
                    add.u64 %rd10, %rd10, %rd11;
                    mul.wide.u32 %rd12, %r28, %r1;
                    cvt.u64.u32 %rd13, %r27;
                    add.u64 %rd12, %rd12, %rd13;
                    selp.b64 %rd14, %rd12, %rd10, %p1;
                    shl.b64 %rd14, %rd14, 2;
                    add.u64 %rd14, %rd14, %rd1;
                    mov.f32 %f1, 0f00000000;
                    @%p3 ld.global.f32 %f1, [%rd14];
                    shl.b32 %r29, %r26, {{log}};
                    add.u32 %r29, %r29, %r25;
                    shl.b32 %r29, %r29, 2;
                    add.u32 %r29, %r29, %r12;
                    st.shared.f32 [%r29], %f1;
                """);
        }

        for (int r = 0; r < loads && packed > 0; r++)
        {
            // Packed B (kk = e / tile, j = e % tile): the word holding b[k, j], its byte / nibble / half, times the scale.
            int shift = packed switch { 1 => 2, 2 => 3, _ => 1 };
            string expand = packed switch
            {
                1 => """
                    and.b32 %r43, %r27, 3;
                    shl.b32 %r43, %r43, 3;
                    bfe.s32 %r42, %r42, %r43, 8;
                    cvt.rn.f32.s32 %f2, %r42;
                    mul.wide.u32 %rd30, %r27, 4;
                    add.u64 %rd30, %rd30, %rd29;
                    @%p3 ld.global.f32 %f3, [%rd30];
                    mul.f32 %f2, %f2, %f3;
                    """,
                2 => """
                    and.b32 %r43, %r27, 7;
                    shl.b32 %r43, %r43, 2;
                    bfe.s32 %r42, %r42, %r43, 4;
                    cvt.rn.f32.s32 %f2, %r42;
                    shr.u32 %r44, %r28, 5;
                    shl.b32 %r45, %r41, 3;
                    mad.lo.u32 %r44, %r44, %r45, %r27;
                    mul.wide.u32 %rd30, %r44, 4;
                    add.u64 %rd30, %rd30, %rd29;
                    @%p3 ld.global.f32 %f3, [%rd30];
                    mul.f32 %f2, %f2, %f3;
                    """,
                _ => """
                    and.b32 %r43, %r27, 1;
                    setp.eq.u32 %p12, %r43, 0;
                    shl.b32 %r44, %r42, 16;
                    and.b32 %r45, %r42, 0xFFFF0000;
                    selp.b32 %r42, %r44, %r45, %p12;
                    mov.b32 %f2, %r42;
                    """,
            };
            s.AppendLine($$"""
                    add.u32 %r20, %r6, {{GemmThreads * r}};
                    shr.u32 %r21, %r20, {{log}};
                    and.b32 %r22, %r20, {{tile - 1}};
                    add.u32 %r27, %r10, %r22;
                    add.u32 %r28, %r11, %r21;
                    setp.lt.u32 %p3, %r27, %r2;
                    setp.lt.u32 %p4, %r28, %r3;
                    and.pred %p3, %p3, %p4;
                    mul.wide.u32 %rd10, %r28, %r41;
                    shr.u32 %r46, %r27, {{shift}};
                    cvt.u64.u32 %rd11, %r46;
                    add.u64 %rd10, %rd10, %rd11;
                    shl.b64 %rd10, %rd10, 2;
                    add.u64 %rd14, %rd10, %rd2;
                    mov.u32 %r42, 0;
                    mov.f32 %f3, 0f3F800000;
                    @%p3 ld.global.u32 %r42, [%rd14];
                    {{expand}}
                    mov.f32 %f1, 0f00000000;
                    @%p3 mov.f32 %f1, %f2;
                    shl.b32 %r29, %r21, {{log}};
                    add.u32 %r29, %r29, %r22;
                    shl.b32 %r29, %r29, 2;
                    add.u32 %r29, %r29, %r13;
                    st.shared.f32 [%r29], %f1;
                """);
        }

        for (int r = 0; r < loads && packed == 0; r++)
        {
            // B: not transposed: (kk = e / tile, j = e % tile) with b[k, j] at k·n + j; transposed: (j = e / 8, kk = e % 8)
            // with b stored [n, K]. Stored as bs[kk, j].
            s.AppendLine($$"""
                    add.u32 %r20, %r6, {{GemmThreads * r}};
                    shr.u32 %r21, %r20, {{log}};
                    and.b32 %r22, %r20, {{tile - 1}};
                    shr.u32 %r23, %r20, 3;
                    and.b32 %r24, %r20, 7;
                    selp.b32 %r25, %r23, %r22, %p2;
                    selp.b32 %r26, %r24, %r21, %p2;
                    add.u32 %r27, %r10, %r25;
                    add.u32 %r28, %r11, %r26;
                    setp.lt.u32 %p3, %r27, %r2;
                    setp.lt.u32 %p4, %r28, %r3;
                    and.pred %p3, %p3, %p4;
                    mul.wide.u32 %rd10, %r28, %r2;
                    cvt.u64.u32 %rd11, %r27;
                    add.u64 %rd10, %rd10, %rd11;
                    mul.wide.u32 %rd12, %r27, %r3;
                    cvt.u64.u32 %rd13, %r28;
                    add.u64 %rd12, %rd12, %rd13;
                    selp.b64 %rd14, %rd12, %rd10, %p2;
                    shl.b64 %rd14, %rd14, 2;
                    add.u64 %rd14, %rd14, %rd2;
                    mov.f32 %f1, 0f00000000;
                    @%p3 ld.global.f32 %f1, [%rd14];
                    shl.b32 %r29, %r26, {{log}};
                    add.u32 %r29, %r29, %r25;
                    shl.b32 %r29, %r29, 2;
                    add.u32 %r29, %r29, %r13;
                    st.shared.f32 [%r29], %f1;
                """);
        }

        s.AppendLine("    bar.sync 0;");
        for (int kk = 0; kk < K; kk++)
        {
            for (int v = 0; v < per / 4; v++)
            {
                s.AppendLine($"    ld.shared.v4.f32 {{%fa{4 * v}, %fa{4 * v + 1}, %fa{4 * v + 2}, %fa{4 * v + 3}}}, [%r31+{kk * tile * 4 + 16 * v}];");
                s.AppendLine($"    ld.shared.v4.f32 {{%fb{4 * v}, %fb{4 * v + 1}, %fb{4 * v + 2}, %fb{4 * v + 3}}}, [%r32+{kk * tile * 4 + 16 * v}];");
            }

            for (int i = 0; i < per; i++)
            {
                for (int j = 0; j < per; j++)
                {
                    s.AppendLine($"    fma.rn.f32 %acc{i * per + j}, %fa{i}, %fb{j}, %acc{i * per + j};");
                }
            }
        }

        s.AppendLine("""
                bar.sync 0;
                add.u32 %r11, %r11, 8;
                bra KLOOP;
            KEND:
            """);
        s.AppendLine($"""
                shl.b32 %r33, %r8, {(int)Math.Log2(per)};
                add.u32 %r33, %r33, %r9;
                shl.b32 %r34, %r7, {(int)Math.Log2(per)};
                add.u32 %r34, %r34, %r10;
            """);
        for (int i = 0; i < per; i++)
        {
            s.AppendLine($"""
                    add.u32 %r35, %r33, {i};
                    setp.ge.u32 %p5, %r35, %r1;
                    @%p5 bra STORED;
                    mul.wide.u32 %rd20, %r35, %r2;
                    cvt.u64.u32 %rd21, %r34;
                    add.u64 %rd20, %rd20, %rd21;
                    shl.b64 %rd20, %rd20, 2;
                    add.u64 %rd20, %rd20, %rd3;
                """);
            for (int j = 0; j < per; j++)
            {
                s.AppendLine($"""
                        add.u32 %r36, %r34, {j};
                        setp.lt.u32 %p6, %r36, %r2;
                        and.pred %p8, %p6, %p7;
                        @%p8 ld.global.f32 %f2, [%rd20+{4 * j}];
                        @%p8 fma.rn.f32 %acc{i * per + j}, %f2, %beta, %acc{i * per + j};
                        @%p6 st.global.f32 [%rd20+{4 * j}], %acc{i * per + j};
                    """);
            }
        }

        s.AppendLine("""
            STORED:
                ret;
            }
            """);
        sb.AppendLine(s.ToString());
    }

    // Attention of one query row over the filled part of a key/value cache (see Backend.AttentionDecode), from its
    // window's start with a window (%r28; the splits share the window's positions) and with soft-capped scores with a
    // cap (%p0 in the loop). One block
    // per row; warp w takes positions w, w + 8, … with the lanes splitting the head dimension (lane + 32·i), keeping a
    // running maximum, softmax sum and weighted value sum (online softmax). The warps' partial results are then
    // combined in shared memory: part[w] = (max, sum, acc[256]).
    // Block y of n takes ⌈length / n⌉ positions, at least `minchunk` (measured per shape: with a short filled length the
    // blocks beyond it find nothing to read and only add an empty part, so a short cache runs on fewer blocks than the
    // split count recorded for a full one).
    // int8: keys and values are an int8 cache (rows of `words` packed words, 4 bytes each, byte d % 4 of word d / 4 for
    // dimension d) with one scale per cached row in kscales/vscales; a lane reads its dimension's byte and scales it.
    // bf16: keys and values are bfloat16 pairs (dimension d is half d % 2 of word d / 2, rows of `words` words); a lane
    // shifts its half into the top of a float.
    private static void AttentionDecode(StringBuilder sb, bool int8 = false, bool bf16 = false)
    {
        const int Stride = (2 + DecodeMaxDim) * 4;
        string stride = int8 || bf16 ? "%s_words" : "%s_dim";
        string laneOffset = bf16
            ? """
              shr.u32 %r17, %lane, 1;
              mul.wide.u32 %rd7, %r17, 4;
              and.b32 %r17, %lane, 1;
              xor.b32 %r17, %r17, 1;
              shl.b32 %r17, %r17, 4;
              """
            : int8
            ? """
              shr.u32 %r17, %lane, 2;
              mul.wide.u32 %rd7, %r17, 4;
              and.b32 %r17, %lane, 3;
              shl.b32 %r17, %r17, 3;
              mul.wide.u32 %rd13, %r13, 4;
              add.u64 %rd14, %b_kscales, %rd13;
              add.u64 %rd15, %b_vscales, %rd13;
              """
            : "mov.u64 %rd7, %rd8;";
        int step = int8 ? 32 : bf16 ? 64 : 128;

        // Element i of this lane (dimension lane + 32·i) of the cached row at %rd{base}, as a float in %f3.
        string Load(int i, string baseRegister, string scaleRegister) => bf16
            ? $"""
              @%p{1 + i} ld.global.u32 %r16, [{baseRegister}+{step * i}];
              @%p{1 + i} shl.b32 %r16, %r16, %r17;
              @%p{1 + i} and.b32 %r16, %r16, 0xFFFF0000;
              @%p{1 + i} mov.b32 %f3, %r16;
              """
            : int8
            ? $"""
              @%p{1 + i} ld.global.u32 %r16, [{baseRegister}+{step * i}];
              @%p{1 + i} bfe.s32 %r16, %r16, %r17, 8;
              @%p{1 + i} cvt.rn.f32.s32 %f3, %r16;
              @%p{1 + i} mul.f32 %f3, %f3, {scaleRegister};
              """
            : $"@%p{1 + i} ld.global.f32 %f3, [{baseRegister}+{step * i}];";
        var b = new StringBuilder();
        b.AppendLine($$"""
            div.u32 %r7, %row, %s_rph;
            rem.u32 %r8, %row, %s_rph;
            rem.u32 %r9, %r8, %s_steps;
            ld.global.f32 %f1, [%b_pos];
            cvt.rzi.u32.f32 %r10, %f1;
            add.u32 %r10, %r10, %r9;
            sub.u32 %r11, %s_cap, 1;
            min.u32 %r10, %r10, %r11;
            add.u32 %r10, %r10, 1;
            {{WindowStartPtx("%r28", "%r10", "%s_window", "%p0")}}
            mov.u32 %r22, %ctaid.y;
            mov.u32 %r23, %nctaid.y;
            add.u32 %r24, %r10, %r23;
            sub.u32 %r24, %r24, %r28;
            sub.u32 %r24, %r24, 1;
            div.u32 %r24, %r24, %r23;
            max.u32 %r24, %r24, %s_minchunk;
            mad.lo.u32 %r25, %r22, %r24, %r28;
            add.u32 %r26, %r25, %r24;
            min.u32 %r10, %r26, %r10;
            mul.lo.u32 %r12, %row, %s_dim;
            mul.wide.u32 %rd1, %r12, 4;
            add.u64 %rd2, %b_q, %rd1;
            add.u64 %rd3, %b_y, %rd1;
            mul.lo.u32 %r13, %r7, %s_cap;
            mul.wide.u32 %rd4, %r13, {{stride}};
            shl.b64 %rd4, %rd4, 2;
            add.u64 %rd5, %b_keys, %rd4;
            add.u64 %rd6, %b_values, %rd4;
            mul.wide.u32 %rd8, %lane, 4;
            add.u64 %rd9, %rd2, %rd8;
            {{laneOffset}}
            """);
        for (int i = 0; i < 8; i++)
        {
            b.AppendLine($"""
                add.u32 %r14, %lane, {32 * i};
                setp.lt.u32 %p{1 + i}, %r14, %s_dim;
                mov.f32 %f{10 + i}, {Zero};
                @%p{1 + i} ld.global.f32 %f{10 + i}, [%rd9+{128 * i}];
                mov.f32 %f{22 + i}, {Zero};
                """);
        }

        b.AppendLine($"""
            mov.f32 %f20, {NegInf};
            mov.f32 %f21, {Zero};
            add.u32 %r15, %r25, %warp;
            setp.gt.f32 %p0, %s_softcap, {Zero};
            DL:
            setp.ge.u32 %p9, %r15, %r10;
            @%p9 bra DL_END;
            mul.wide.u32 %rd10, %r15, {stride};
            shl.b64 %rd10, %rd10, 2;
            add.u64 %rd10, %rd10, %rd7;
            add.u64 %rd11, %rd5, %rd10;
            add.u64 %rd12, %rd6, %rd10;
            mov.f32 %f2, {Zero};
            """);
        if (int8)
        {
            b.AppendLine("""
                mul.wide.u32 %rd13, %r15, 4;
                add.u64 %rd13, %rd13, %rd14;
                ld.global.f32 %f7, [%rd13];
                mul.wide.u32 %rd13, %r15, 4;
                add.u64 %rd13, %rd13, %rd15;
                ld.global.f32 %f8, [%rd13];
                """);
        }
        for (int i = 0; i < 8; i++)
        {
            b.AppendLine(Load(i, "%rd11", "%f7"));
            b.AppendLine($"@%p{1 + i} fma.rn.f32 %f2, %f{10 + i}, %f3, %f2;");
        }

        foreach (int offset in new[] { 16, 8, 4, 2, 1 })
        {
            b.AppendLine($"shfl.sync.bfly.b32 %f3, %f2, {offset}, 31, 0xffffffff;");
            b.AppendLine("add.f32 %f2, %f2, %f3;");
        }

        b.AppendLine($"""
            mul.f32 %f2, %f2, %s_scale;
            {SoftcapPtx("%f2", "%s_softcap", "%p0", "%f18")}
            max.f32 %f4, %f20, %f2;
            sub.f32 %f5, %f20, %f4;
            mul.f32 %f5, %f5, {Log2E};
            ex2.approx.ftz.f32 %f5, %f5;
            sub.f32 %f6, %f2, %f4;
            mul.f32 %f6, %f6, {Log2E};
            ex2.approx.ftz.f32 %f6, %f6;
            fma.rn.f32 %f21, %f21, %f5, %f6;
            """);
        for (int i = 0; i < 8; i++)
        {
            b.AppendLine($"mul.f32 %f{22 + i}, %f{22 + i}, %f5;");
            b.AppendLine(Load(i, "%rd12", "%f8"));
            b.AppendLine($"@%p{1 + i} fma.rn.f32 %f{22 + i}, %f6, %f3, %f{22 + i};");
        }

        b.AppendLine($"""
            mov.f32 %f20, %f4;
            add.u32 %r15, %r15, %nwarps;
            bra DL;
            DL_END:
            mul.lo.u32 %r16, %warp, {Stride};
            add.u32 %r16, %r16, %spart;
            setp.eq.u32 %p10, %lane, 0;
            @%p10 st.shared.f32 [%r16], %f20;
            @%p10 st.shared.f32 [%r16+4], %f21;
            shl.b32 %r17, %lane, 2;
            add.u32 %r17, %r17, %r16;
            """);
        for (int i = 0; i < 8; i++)
        {
            b.AppendLine($"@%p{1 + i} st.shared.f32 [%r17+{8 + 128 * i}], %f{22 + i};");
        }

        b.AppendLine($"""
            bar.sync 0;
            mov.f32 %f7, {NegInf};
            mov.u32 %r18, 0;
            CM:
            setp.ge.u32 %p11, %r18, %nwarps;
            @%p11 bra CM_END;
            mul.lo.u32 %r19, %r18, {Stride};
            add.u32 %r19, %r19, %spart;
            ld.shared.f32 %f8, [%r19];
            max.f32 %f7, %f7, %f8;
            add.u32 %r18, %r18, 1;
            bra CM;
            CM_END:
            mov.f32 %f28, %f7;
            setp.eq.f32 %p13, %f7, {NegInf};
            @%p13 mov.f32 %f7, {Zero};
            mov.f32 %f9, {Zero};
            mov.u32 %r18, 0;
            CL:
            setp.ge.u32 %p11, %r18, %nwarps;
            @%p11 bra CL_END;
            mul.lo.u32 %r19, %r18, {Stride};
            add.u32 %r19, %r19, %spart;
            ld.shared.f32 %f8, [%r19];
            sub.f32 %f8, %f8, %f7;
            mul.f32 %f8, %f8, {Log2E};
            ex2.approx.ftz.f32 %f8, %f8;
            ld.shared.f32 %f31, [%r19+4];
            fma.rn.f32 %f9, %f31, %f8, %f9;
            add.u32 %r18, %r18, 1;
            bra CL;
            CL_END:
            setp.gt.u32 %p14, %r23, 1;
            mad.lo.u32 %r26, %row, %r23, %r22;
            add.u32 %r27, %s_dim, 2;
            mul.lo.u32 %r26, %r26, %r27;
            mul.wide.u32 %rd10, %r26, 4;
            add.u64 %rd10, %rd10, %b_part;
            setp.eq.u32 %p15, %tx, 0;
            and.pred %p15, %p15, %p14;
            @%p15 st.global.f32 [%rd10], %f28;
            @%p15 st.global.f32 [%rd10+4], %f9;
            @!%p14 rcp.rn.f32 %f9, %f9;
            @%p14 mov.f32 %f9, 0f3F800000;
            mov.u32 %r20, %tx;
            CO:
            setp.ge.u32 %p12, %r20, %s_dim;
            @%p12 bra CO_END;
            mov.f32 %f30, {Zero};
            mov.u32 %r18, 0;
            COW:
            setp.ge.u32 %p11, %r18, %nwarps;
            @%p11 bra COW_END;
            mul.lo.u32 %r19, %r18, {Stride};
            add.u32 %r19, %r19, %spart;
            ld.shared.f32 %f8, [%r19];
            sub.f32 %f8, %f8, %f7;
            mul.f32 %f8, %f8, {Log2E};
            ex2.approx.ftz.f32 %f8, %f8;
            shl.b32 %r21, %r20, 2;
            add.u32 %r21, %r21, %r19;
            ld.shared.f32 %f31, [%r21+8];
            fma.rn.f32 %f30, %f31, %f8, %f30;
            add.u32 %r18, %r18, 1;
            bra COW;
            COW_END:
            mul.f32 %f30, %f30, %f9;
            mul.wide.u32 %rd13, %r20, 4;
            @%p14 add.u64 %rd13, %rd13, %rd10;
            @%p14 add.u64 %rd13, %rd13, 8;
            @!%p14 add.u64 %rd13, %rd13, %rd3;
            st.global.f32 [%rd13], %f30;
            add.u32 %r20, %r20, %nt;
            bra CO;
            CO_END:
            """);

        // With several splits, the last block of this row to finish (counted per row, reset afterwards) merges the
        // splits' partial results: part[row, split] = (max, sum, acc[dim]) over its chunk of cached positions (max -inf
        // when empty); y[row] = Σ acc·e^(max - M) / Σ sum·e^(max - M), M the largest max. Same launch, same order of sums.
        b.AppendLine($"""
            @!%p14 bra DONE;
            membar.gl;
            bar.sync 0;
            mul.wide.u32 %rd1, %row, 4;
            add.u64 %rd1, %rd1, %b_counters;
            setp.eq.u32 %p15, %tx, 0;
            mov.u32 %r5, 0;
            @%p15 atom.global.add.u32 %r5, [%rd1], 1;
            sub.u32 %r6, %r23, 1;
            setp.eq.u32 %p1, %r5, %r6;
            selp.u32 %r7, 1, 0, %p1;
            @%p15 st.shared.u32 [%sbi], %r7;
            bar.sync 0;
            ld.shared.u32 %r7, [%sbi];
            setp.eq.u32 %p1, %r7, 0;
            @%p1 bra DONE;
            membar.gl;
            @%p15 st.global.u32 [%rd1], 0;
            mul.lo.u32 %r5, %row, %r23;
            add.u32 %r6, %s_dim, 2;
            mul.lo.u32 %r7, %r5, %r6;
            mul.wide.u32 %rd1, %r7, 4;
            add.u64 %rd1, %rd1, %b_part;
            mul.wide.u32 %rd2, %r6, 4;
            mov.f32 %f1, {NegInf};
            mov.u32 %r8, 0;
            mov.u64 %rd3, %rd1;
            FM:
            setp.ge.u32 %p1, %r8, %r23;
            @%p1 bra FM_END;
            ld.global.cg.f32 %f2, [%rd3];
            max.f32 %f1, %f1, %f2;
            add.u64 %rd3, %rd3, %rd2;
            add.u32 %r8, %r8, 1;
            bra FM;
            FM_END:
            mov.f32 %f3, {Zero};
            mov.u32 %r8, 0;
            mov.u64 %rd3, %rd1;
            FS:
            setp.ge.u32 %p1, %r8, %r23;
            @%p1 bra FS_END;
            ld.global.cg.f32 %f2, [%rd3];
            sub.f32 %f2, %f2, %f1;
            mul.f32 %f2, %f2, {Log2E};
            ex2.approx.ftz.f32 %f2, %f2;
            ld.global.cg.f32 %f4, [%rd3+4];
            fma.rn.f32 %f3, %f4, %f2, %f3;
            add.u64 %rd3, %rd3, %rd2;
            add.u32 %r8, %r8, 1;
            bra FS;
            FS_END:
            rcp.rn.f32 %f3, %f3;
            mul.lo.u32 %r10, %row, %s_dim;
            mul.wide.u32 %rd6, %r10, 4;
            add.u64 %rd6, %rd6, %b_y;
            mov.u32 %r9, %tx;
            FD:
            setp.ge.u32 %p2, %r9, %s_dim;
            @%p2 bra FD_END;
            mov.f32 %f5, {Zero};
            mov.u32 %r8, 0;
            mov.u64 %rd3, %rd1;
            mul.wide.u32 %rd4, %r9, 4;
            FL:
            setp.ge.u32 %p1, %r8, %r23;
            @%p1 bra FL_END;
            ld.global.cg.f32 %f2, [%rd3];
            sub.f32 %f2, %f2, %f1;
            mul.f32 %f2, %f2, {Log2E};
            ex2.approx.ftz.f32 %f2, %f2;
            add.u64 %rd5, %rd3, %rd4;
            ld.global.cg.f32 %f6, [%rd5+8];
            fma.rn.f32 %f5, %f6, %f2, %f5;
            add.u64 %rd3, %rd3, %rd2;
            add.u32 %r8, %r8, 1;
            bra FL;
            FL_END:
            mul.f32 %f5, %f5, %f3;
            add.u64 %rd7, %rd6, %rd4;
            st.global.f32 [%rd7], %f5;
            add.u32 %r9, %r9, %nt;
            bra FD;
            FD_END:
            """);
        if (bf16)
        {
            RowBlock(sb, "attention_decode_bf16", ["q", "keys", "values", "pos", "y", "part", "counters"],
                [("u32", "rph"), ("u32", "steps"), ("u32", "cap"), ("u32", "dim"), ("f32", "scale"), ("u32", "words"), ("u32", "minchunk"), ("u32", "window"), ("f32", "softcap")],
                b.ToString(), sharedFloats: RowThreads / 32 * (2 + DecodeMaxDim));
        }
        else if (int8)
        {
            RowBlock(sb, "attention_decode_int8", ["q", "keys", "values", "kscales", "vscales", "pos", "y", "part", "counters"],
                [("u32", "rph"), ("u32", "steps"), ("u32", "cap"), ("u32", "dim"), ("f32", "scale"), ("u32", "words"), ("u32", "minchunk"), ("u32", "window"), ("f32", "softcap")],
                b.ToString(), sharedFloats: RowThreads / 32 * (2 + DecodeMaxDim));
        }
        else
        {
            RowBlock(sb, "attention_decode_f32", ["q", "keys", "values", "pos", "y", "part", "counters"],
                [("u32", "rph"), ("u32", "steps"), ("u32", "cap"), ("u32", "dim"), ("f32", "scale"), ("u32", "minchunk"), ("u32", "window"), ("f32", "softcap")],
                b.ToString(), sharedFloats: RowThreads / 32 * (2 + DecodeMaxDim));

        }
    }

    /// <summary>Threads of a <c>gemv_nn_f32</c> block: 32 columns × 32 slices of k.</summary>
    public const int GemvThreads = 1024;

    // c[b][r, j] = Σ_k a[b][r, k] · w[b][k, j] (+ beta · c) for m ≤ 8 rows. Block: 32 columns × 32 slices of k (slice s
    // takes k = s, s + 32, …, four at a time so several loads are in flight); each thread keeps 8 row sums, the slices
    // are added in shared memory and warp r writes row r. Grid: x = ⌈n / 32⌉, z = batch. Reads of w are coalesced.
    //
    // gemv_multi_f32: up to three products sharing the input a [m, k] (the query/key/value or gate/up projections):
    // y_j = a · w_j (+ bias_j) with w_j [k, n_j]; grid y selects j (blocks beyond n_j return at once).
    private static void GemvNN(StringBuilder sb, bool multi = false)
    {
        var s = new StringBuilder();
        if (multi)
        {
            s.AppendLine("""
                .visible .entry gemv_multi_f32(
                    .param .u64 p_a, .param .u32 p_m, .param .u32 p_k,
                    .param .u64 p_w0, .param .u64 p_bias0, .param .u64 p_y0, .param .u32 p_n0,
                    .param .u64 p_w1, .param .u64 p_bias1, .param .u64 p_y1, .param .u32 p_n1,
                    .param .u64 p_w2, .param .u64 p_bias2, .param .u64 p_y2, .param .u32 p_n2
                )
                {
                    .reg .pred %p<16>;
                    .reg .f32 %f<32>;
                    .reg .b32 %r<24>;
                    .reg .b64 %rd<24>;
                    .reg .b64 %w<3>;
                    .reg .b64 %bias<3>;
                    .reg .b64 %y<3>;
                    .reg .b32 %nn<3>;
                    .shared .align 4 .f32 gemv_multi_part[8192];
                    ld.param.u64 %rd1, [p_a];
                    cvta.to.global.u64 %rd1, %rd1;
                    ld.param.u32 %r1, [p_m];
                    ld.param.u32 %r3, [p_k];
                """);
            for (int j = 0; j < 3; j++)
            {
                s.AppendLine($"""
                        ld.param.u64 %w{j}, [p_w{j}];
                        ld.param.u64 %bias{j}, [p_bias{j}];
                        ld.param.u64 %y{j}, [p_y{j}];
                        ld.param.u32 %nn{j}, [p_n{j}];
                    """);
            }

            // Select this block's matrix; the bias pointer stays 0 when the layer has none.
            s.AppendLine("""
                    mov.u32 %r4, %ctaid.y;
                    setp.eq.u32 %p13, %r4, 1;
                    setp.eq.u32 %p14, %r4, 2;
                    selp.b64 %rd2, %w1, %w0, %p13;
                    selp.b64 %rd2, %w2, %rd2, %p14;
                    selp.b64 %rd15, %bias1, %bias0, %p13;
                    selp.b64 %rd15, %bias2, %rd15, %p14;
                    selp.b64 %rd3, %y1, %y0, %p13;
                    selp.b64 %rd3, %y2, %rd3, %p14;
                    selp.b32 %r2, %nn1, %nn0, %p13;
                    selp.b32 %r2, %nn2, %r2, %p14;
                    mov.u32 %r8, %ctaid.x;
                    shl.b32 %r8, %r8, 5;
                    setp.ge.u32 %p15, %r8, %r2;
                    @%p15 bra DONE;
                    cvta.to.global.u64 %rd2, %rd2;
                    cvta.to.global.u64 %rd3, %rd3;
                    setp.ne.u64 %p13, %rd15, 0;
                    @%p13 cvta.to.global.u64 %rd15, %rd15;
                    mov.f32 %f20, 0f00000000;
                    mov.u32 %r5, %tid.x;
                    and.b32 %r6, %r5, 31;
                    shr.u32 %r7, %r5, 5;
                    add.u32 %r8, %r8, %r6;
                    setp.lt.u32 %p9, %r8, %r2;
                    mul.wide.u32 %rd12, %r3, 4;
                    mul.wide.u32 %rd14, %r2, 128;
                    cvt.u64.u32 %rd10, %r8;
                """);
        }
        else
        {
            s.AppendLine("""
                .visible .entry gemv_nn_f32(
                    .param .u64 p_a, .param .u64 p_b, .param .u64 p_c,
                    .param .u32 p_m, .param .u32 p_n, .param .u32 p_k, .param .f32 p_beta,
                    .param .u64 p_sa, .param .u64 p_sb, .param .u64 p_sc
                )
                {
                    .reg .pred %p<16>;
                    .reg .f32 %f<32>;
                    .reg .b32 %r<24>;
                    .reg .b64 %rd<24>;
                    .shared .align 4 .f32 gemv_nn_part[8192];
                    ld.param.u64 %rd1, [p_a];
                    ld.param.u64 %rd2, [p_b];
                    ld.param.u64 %rd3, [p_c];
                    cvta.to.global.u64 %rd1, %rd1;
                    cvta.to.global.u64 %rd2, %rd2;
                    cvta.to.global.u64 %rd3, %rd3;
                    ld.param.u32 %r1, [p_m];
                    ld.param.u32 %r2, [p_n];
                    ld.param.u32 %r3, [p_k];
                    ld.param.f32 %f20, [p_beta];
                    ld.param.u64 %rd4, [p_sa];
                    ld.param.u64 %rd5, [p_sb];
                    ld.param.u64 %rd6, [p_sc];
                    mov.u32 %r4, %ctaid.z;
                    cvt.u64.u32 %rd7, %r4;
                    mul.lo.u64 %rd8, %rd7, %rd4;
                    shl.b64 %rd8, %rd8, 2;
                    add.u64 %rd1, %rd1, %rd8;
                    mul.lo.u64 %rd8, %rd7, %rd5;
                    shl.b64 %rd8, %rd8, 2;
                    add.u64 %rd2, %rd2, %rd8;
                    mul.lo.u64 %rd8, %rd7, %rd6;
                    shl.b64 %rd8, %rd8, 2;
                    add.u64 %rd3, %rd3, %rd8;
                    mov.u32 %r5, %tid.x;
                    and.b32 %r6, %r5, 31;
                    shr.u32 %r7, %r5, 5;
                    mov.u32 %r8, %ctaid.x;
                    shl.b32 %r8, %r8, 5;
                    add.u32 %r8, %r8, %r6;
                    setp.lt.u32 %p9, %r8, %r2;
                    mul.wide.u32 %rd12, %r3, 4;
                    mul.wide.u32 %rd14, %r2, 128;
                    cvt.u64.u32 %rd10, %r8;
                """);
        }

        for (int r = 0; r < GemvRows; r++)
        {
            s.AppendLine($"    mov.f32 %f{r}, 0f00000000;");
            s.AppendLine($"    setp.lt.u32 %p{r}, {r}, %r1;");
        }

        // Four k at a time: kk, kk + 32, kk + 64, kk + 96 while kk + 96 < k.
        s.AppendLine("""
                mov.u32 %r9, %r7;
            K4:
                add.u32 %r14, %r9, 96;
                setp.ge.u32 %p10, %r14, %r3;
                @%p10 bra K1;
                mul.wide.u32 %rd9, %r9, %r2;
                add.u64 %rd9, %rd9, %rd10;
                shl.b64 %rd9, %rd9, 2;
                add.u64 %rd9, %rd9, %rd2;
                mov.f32 %f8, 0f00000000;
                mov.f32 %f10, 0f00000000;
                mov.f32 %f11, 0f00000000;
                mov.f32 %f12, 0f00000000;
                @%p9 ld.global.f32 %f8, [%rd9];
                add.u64 %rd9, %rd9, %rd14;
                @%p9 ld.global.f32 %f10, [%rd9];
                add.u64 %rd9, %rd9, %rd14;
                @%p9 ld.global.f32 %f11, [%rd9];
                add.u64 %rd9, %rd9, %rd14;
                @%p9 ld.global.f32 %f12, [%rd9];
                mul.wide.u32 %rd11, %r9, 4;
                add.u64 %rd11, %rd11, %rd1;
            """);
        for (int r = 0; r < GemvRows; r++)
        {
            s.AppendLine($"""
                    @!%p{r} bra K4_ROWS_DONE;
                    ld.global.f32 %f13, [%rd11];
                    ld.global.f32 %f14, [%rd11+128];
                    ld.global.f32 %f15, [%rd11+256];
                    ld.global.f32 %f16, [%rd11+384];
                    fma.rn.f32 %f{r}, %f13, %f8, %f{r};
                    fma.rn.f32 %f{r}, %f14, %f10, %f{r};
                    fma.rn.f32 %f{r}, %f15, %f11, %f{r};
                    fma.rn.f32 %f{r}, %f16, %f12, %f{r};
                    add.u64 %rd11, %rd11, %rd12;
                """);
        }

        s.AppendLine("""
            K4_ROWS_DONE:
                add.u32 %r9, %r9, 128;
                bra K4;
            K1:
                setp.ge.u32 %p10, %r9, %r3;
                @%p10 bra KEND;
                mov.f32 %f8, 0f00000000;
                mul.wide.u32 %rd9, %r9, %r2;
                add.u64 %rd9, %rd9, %rd10;
                shl.b64 %rd9, %rd9, 2;
                add.u64 %rd9, %rd9, %rd2;
                @%p9 ld.global.f32 %f8, [%rd9];
                mul.wide.u32 %rd11, %r9, 4;
                add.u64 %rd11, %rd11, %rd1;
            """);
        for (int r = 0; r < GemvRows; r++)
        {
            s.AppendLine($"    @%p{r} ld.global.f32 %f9, [%rd11];");
            s.AppendLine($"    @%p{r} fma.rn.f32 %f{r}, %f9, %f8, %f{r};");
            s.AppendLine("    add.u64 %rd11, %rd11, %rd12;");
        }

        // part[slice][row][lane]: slice stride 1024 bytes, row stride 128 bytes.
        string part = multi ? "gemv_multi_part" : "gemv_nn_part";
        s.AppendLine($$"""
                add.u32 %r9, %r9, 32;
                bra K1;
            KEND:
                mov.u32 %r10, {{part}};
                shl.b32 %r11, %r7, 10;
                shl.b32 %r12, %r6, 2;
                add.u32 %r11, %r11, %r12;
                add.u32 %r11, %r11, %r10;
            """);
        for (int r = 0; r < GemvRows; r++)
        {
            s.AppendLine($"    st.shared.f32 [%r11+{r * 128}], %f{r};");
        }

        s.AppendLine("""
                bar.sync 0;
                setp.ge.u32 %p11, %r7, %r1;
                @%p11 bra DONE;
                @!%p9 bra DONE;
                shl.b32 %r13, %r7, 7;
                add.u32 %r13, %r13, %r12;
                add.u32 %r13, %r13, %r10;
                mov.f32 %f17, 0f00000000;
            """);
        for (int slice = 0; slice < 32; slice++)
        {
            s.AppendLine($"    ld.shared.f32 %f18, [%r13+{slice * 1024}];");
            s.AppendLine("    add.f32 %f17, %f17, %f18;");
        }

        s.AppendLine("""
                mul.wide.u32 %rd13, %r7, %r2;
                add.u64 %rd13, %rd13, %rd10;
                shl.b64 %rd13, %rd13, 2;
                add.u64 %rd13, %rd13, %rd3;
                setp.eq.f32 %p12, %f20, 0f00000000;
                @%p12 bra STORE;
                ld.global.f32 %f19, [%rd13];
                fma.rn.f32 %f17, %f19, %f20, %f17;
            STORE:
            """);
        if (multi)
        {
            s.AppendLine("""
                    setp.ne.u64 %p13, %rd15, 0;
                    @!%p13 bra NO_BIAS;
                    mul.wide.u32 %rd16, %r8, 4;
                    add.u64 %rd16, %rd16, %rd15;
                    ld.global.f32 %f19, [%rd16];
                    add.f32 %f17, %f17, %f19;
                NO_BIAS:
                """);
        }

        s.AppendLine("""
                st.global.f32 [%rd13], %f17;
            DONE:
                ret;
            }
            """);
        sb.AppendLine(s.ToString());
    }

    // c[b][r, j] = Σ_k a[b][r, k] · w[b][j, k] (+ beta · c), w given as [n, k] (a product with a transposed matrix,
    // such as queries times keys), for m ≤ 8 rows. One warp per column: lanes stride over k (coalesced), then a warp
    // reduction per row. Grid: x = ⌈n / 8⌉ (8 warps per block), z = batch.
    private static void GemvNT(StringBuilder sb)
    {
        var s = new StringBuilder();
        s.AppendLine("""
            .visible .entry gemv_nt_f32(
                .param .u64 p_a, .param .u64 p_b, .param .u64 p_c,
                .param .u32 p_m, .param .u32 p_n, .param .u32 p_k, .param .f32 p_beta,
                .param .u64 p_sa, .param .u64 p_sb, .param .u64 p_sc
            )
            {
                .reg .pred %p<16>;
                .reg .f32 %f<24>;
                .reg .b32 %r<24>;
                .reg .b64 %rd<24>;
                ld.param.u64 %rd1, [p_a];
                ld.param.u64 %rd2, [p_b];
                ld.param.u64 %rd3, [p_c];
                cvta.to.global.u64 %rd1, %rd1;
                cvta.to.global.u64 %rd2, %rd2;
                cvta.to.global.u64 %rd3, %rd3;
                ld.param.u32 %r1, [p_m];
                ld.param.u32 %r2, [p_n];
                ld.param.u32 %r3, [p_k];
                ld.param.f32 %f20, [p_beta];
                ld.param.u64 %rd4, [p_sa];
                ld.param.u64 %rd5, [p_sb];
                ld.param.u64 %rd6, [p_sc];
                mov.u32 %r4, %ctaid.z;
                cvt.u64.u32 %rd7, %r4;
                mul.lo.u64 %rd8, %rd7, %rd4;
                shl.b64 %rd8, %rd8, 2;
                add.u64 %rd1, %rd1, %rd8;
                mul.lo.u64 %rd8, %rd7, %rd5;
                shl.b64 %rd8, %rd8, 2;
                add.u64 %rd2, %rd2, %rd8;
                mul.lo.u64 %rd8, %rd7, %rd6;
                shl.b64 %rd8, %rd8, 2;
                add.u64 %rd3, %rd3, %rd8;
                mov.u32 %r5, %tid.x;
                and.b32 %r6, %r5, 31;
                shr.u32 %r7, %r5, 5;
                mov.u32 %r8, %ctaid.x;
                shl.b32 %r8, %r8, 3;
                add.u32 %r8, %r8, %r7;
                setp.ge.u32 %p9, %r8, %r2;
                @%p9 bra DONE;
                mul.wide.u32 %rd9, %r8, %r3;
                shl.b64 %rd9, %rd9, 2;
                add.u64 %rd9, %rd9, %rd2;
            """);
        for (int r = 0; r < GemvRows; r++)
        {
            s.AppendLine($"    mov.f32 %f{r}, 0f00000000;");
            s.AppendLine($"    setp.lt.u32 %p{r}, {r}, %r1;");
        }

        s.AppendLine("""
                mov.u32 %r9, %r6;
                mul.wide.u32 %rd12, %r3, 4;
            KLOOP:
                setp.ge.u32 %p10, %r9, %r3;
                @%p10 bra KEND;
                mul.wide.u32 %rd10, %r9, 4;
                add.u64 %rd11, %rd10, %rd9;
                ld.global.f32 %f8, [%rd11];
                add.u64 %rd11, %rd10, %rd1;
            """);
        for (int r = 0; r < GemvRows; r++)
        {
            s.AppendLine($"    @%p{r} ld.global.f32 %f9, [%rd11];");
            s.AppendLine($"    @%p{r} fma.rn.f32 %f{r}, %f9, %f8, %f{r};");
            s.AppendLine("    add.u64 %rd11, %rd11, %rd12;");
        }

        s.AppendLine("""
                add.u32 %r9, %r9, 32;
                bra KLOOP;
            KEND:
            """);
        for (int r = 0; r < GemvRows; r++)
        {
            foreach (int offset in new[] { 16, 8, 4, 2, 1 })
            {
                s.AppendLine($"    shfl.sync.bfly.b32 %f9, %f{r}, {offset}, 31, 0xffffffff;");
                s.AppendLine($"    add.f32 %f{r}, %f{r}, %f9;");
            }
        }

        s.AppendLine("""
                setp.ne.u32 %p11, %r6, 0;
                @%p11 bra DONE;
                setp.ne.f32 %p12, %f20, 0f00000000;
                cvt.u64.u32 %rd13, %r8;
                shl.b64 %rd13, %rd13, 2;
                add.u64 %rd13, %rd13, %rd3;
                mul.wide.u32 %rd14, %r2, 4;
            """);
        for (int r = 0; r < GemvRows; r++)
        {
            s.AppendLine($"""
                    @!%p{r} bra DONE;
                    @%p12 ld.global.f32 %f10, [%rd13];
                    @%p12 fma.rn.f32 %f{r}, %f10, %f20, %f{r};
                    st.global.f32 [%rd13], %f{r};
                    add.u64 %rd13, %rd13, %rd14;
                """);
        }

        s.AppendLine("""
            DONE:
                ret;
            }
            """);
        sb.AppendLine(s.ToString());
    }
}
