// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;

namespace Idrak.Gpu.Cuda;

// PTX for int8 weight-only quantization. Weights q[k, n] are signed bytes packed four per 32-bit word along n (each
// row padded to n4 = ceil(n / 4) words), with one float scale per column: w[k, j] = q[k, j] · scale[j].
internal static partial class PtxKernels
{
    public static readonly string[] QuantizedNames = ["int8_matmul_f32", "int8_dequant_f32", "kv_write_int8", "attn_scores_int8", "attn_context_int8", "int8_gemv_f32", "int8_gemv_finish_f32", "bf16_gemv_f32", "bf16_dequant_f32", "f16_dequant_f32", "int4_gemv_f32", "int4_dequant_f32", "int8_gemv_multi_f32", "bf16_gemv_multi_f32", "int4_gemv_multi_f32", "int8_gemv_silu_f32", "bf16_gemv_silu_f32", "int4_gemv_silu_f32", "int8_gemv_gelu_f32", "bf16_gemv_gelu_f32", "int4_gemv_gelu_f32", "int8_gemv_addnorm_f32", "bf16_gemv_addnorm_f32", "int4_gemv_addnorm_f32", "int8_gemv_multi_act_f32", "bf16_gemv_multi_act_f32", "int4_gemv_multi_act_f32", "int8_gemv_normaddnorm_f32", "bf16_gemv_normaddnorm_f32", "int4_gemv_normaddnorm_f32",
        "int8_gemv_r1_f32", "int8_gemv_r2_f32", "int8_gemv_r4_f32", "int4_gemv_r1_f32", "int4_gemv_r2_f32", "int4_gemv_r4_f32",
        "bf16_gemv_r1_f32", "bf16_gemv_r2_f32", "bf16_gemv_r4_f32"];

    /// <summary>
    /// Row counts the plain few-row products are also compiled for (<c>int8_gemv_r1_f32</c> …, by format int8, int4,
    /// bfloat16 and then these counts in <see cref="QuantizedNames"/>): m rows take the first at least m, more than the last
    /// the <see cref="GemvRows"/>-row kernel. Each gives the eight-row kernel's results bit for bit. A span over constant
    /// data, not a static array: Source is built during static initialization, which may precede this file's fields.
    /// </summary>
    public static ReadOnlySpan<int> GemvRowVariants => [1, 2, 4];

    /// <summary>Rows of an int4 weight matrix that share one scale per column.</summary>
    public const int Int4Group = 32;

    /// <summary>Threads of an <c>int8_gemv_f32</c> block: 32 packed words (128 columns) × 16 slices of k.</summary>
    public const int Int8GemvThreads = 512;

    private static void BuildQuantized(StringBuilder sb)
    {
        Int8Gemv(sb);
        Int8Gemv(sb, bf16: true);
        Int8Gemv(sb, int4: true);
        foreach (int gated in new[] { 1, 2 })
        {
            Int8Gemv(sb, gated: gated);
            Int8Gemv(sb, bf16: true, gated: gated);
            Int8Gemv(sb, int4: true, gated: gated);
        }

        Int8Gemv(sb, multi: true);
        Int8Gemv(sb, bf16: true, multi: true);
        Int8Gemv(sb, int4: true, multi: true);
        Int8Gemv(sb, addNorm: true);
        Int8Gemv(sb, multi: true, gatedPair: true);
        Int8Gemv(sb, bf16: true, multi: true, gatedPair: true);
        Int8Gemv(sb, int4: true, multi: true, gatedPair: true);
        Int8Gemv(sb, bf16: true, addNorm: true);
        Int8Gemv(sb, int4: true, addNorm: true);
        Int8Gemv(sb, addNorm: true, normAdd: true);
        Int8Gemv(sb, bf16: true, addNorm: true, normAdd: true);
        Int8Gemv(sb, int4: true, addNorm: true, normAdd: true);
        foreach (int rows in GemvRowVariants)
        {
            Int8Gemv(sb, rows: rows);
            Int8Gemv(sb, int4: true, rows: rows);
            Int8Gemv(sb, bf16: true, rows: rows);
        }

        // w[r, 8c + j] = nibble j of packed word (r, c) · s[r / Int4Group, 8c + j]: one thread per word.
        var dequant = new StringBuilder("""
            ld.global.u32 %r5, [%a_q];
            div.u32 %r6, %i, %s_words;
            rem.u32 %r7, %i, %s_words;
            shl.b32 %r8, %r7, 3;
            shr.u32 %r9, %r6, 5;
            shl.b32 %r10, %s_words, 3;
            mad.lo.u32 %r9, %r9, %r10, %r8;
            mul.wide.u32 %rd2, %r9, 4;
            add.u64 %rd2, %rd2, %b_s;
            mad.lo.u32 %r11, %r6, %s_cols, %r8;
            mul.wide.u32 %rd1, %r11, 4;
            add.u64 %rd1, %rd1, %b_w;
            """);
        for (int c = 0; c < 8; c++)
        {
            dequant.AppendLine(CultureInfo.InvariantCulture, $"""
                add.u32 %r12, %r8, {c};
                setp.ge.u32 %p1, %r12, %s_cols;
                @%p1 bra DONE;
                bfe.s32 %r13, %r5, {4 * c}, 4;
                cvt.rn.f32.s32 %f1, %r13;
                ld.global.f32 %f2, [%rd2+{4 * c}];
                mul.f32 %f1, %f1, %f2;
                st.global.f32 [%rd1+{4 * c}], %f1;
                """);
        }

        Elementwise(sb, "int4_dequant_f32", ["q", "s", "w"], [("u32", "words"), ("u32", "cols")], dequant.ToString());

        // w[r, 2c], w[r, 2c + 1] = the bfloat16 pair in packed word (r, c) (words = ⌈cols / 2⌉ per row): one thread per word.
        Elementwise(sb, "bf16_dequant_f32", ["q", "w"], [("u32", "words"), ("u32", "cols")], """
            ld.global.u32 %r5, [%a_q];
            div.u32 %r6, %i, %s_words;
            rem.u32 %r7, %i, %s_words;
            shl.b32 %r8, %r7, 1;
            mad.lo.u32 %r9, %r6, %s_cols, %r8;
            mul.wide.u32 %rd1, %r9, 4;
            add.u64 %rd1, %rd1, %b_w;
            shl.b32 %r10, %r5, 16;
            mov.b32 %f1, %r10;
            st.global.f32 [%rd1], %f1;
            add.u32 %r11, %r8, 1;
            setp.ge.u32 %p1, %r11, %s_cols;
            @%p1 bra DONE;
            and.b32 %r10, %r5, 0xFFFF0000;
            mov.b32 %f2, %r10;
            st.global.f32 [%rd1+4], %f2;
            """);

        // w[r, 2c], w[r, 2c + 1] = the IEEE half pair in packed word (r, c) widened (cvt.f32.f16: exact, subnormals kept).
        // A NaN half widens to the float NaN with its payload as it is (sign | 0x7F800000 | mantissa << 13), as the CPU's
        // widening (HalfFloats) and Vulkan's do, so every device agrees: cvt.f32.f16 gives one canonical NaN for all of them,
        // and .NET's (float)Half quiets a signalling one (sets the mantissa's top bit), so neither is followed.
        static string KeepNaN(string f) => $"""
            and.b32 %r13, %r12, 0x7C00;
            and.b32 %r14, %r12, 0x3FF;
            setp.eq.u32 %p2, %r13, 0x7C00;
            setp.ne.and.u32 %p2, %r14, 0, %p2;
            and.b32 %r15, %r12, 0x8000;
            shl.b32 %r15, %r15, 16;
            shl.b32 %r14, %r14, 13;
            or.b32 %r15, %r15, %r14;
            or.b32 %r15, %r15, 0x7F800000;
            @%p2 mov.b32 {f}, %r15;
            """;
        Elementwise(sb, "f16_dequant_f32", ["q", "w"], [("u32", "words"), ("u32", "cols")], """
            {
            .reg .b16 %hl, %hh;
            ld.global.u32 %r5, [%a_q];
            div.u32 %r6, %i, %s_words;
            rem.u32 %r7, %i, %s_words;
            shl.b32 %r8, %r7, 1;
            mad.lo.u32 %r9, %r6, %s_cols, %r8;
            mul.wide.u32 %rd1, %r9, 4;
            add.u64 %rd1, %rd1, %b_w;
            mov.b32 {%hl, %hh}, %r5;
            cvt.f32.f16 %f1, %hl;
            and.b32 %r12, %r5, 0xFFFF;
            NAN_LOW
            st.global.f32 [%rd1], %f1;
            add.u32 %r11, %r8, 1;
            setp.ge.u32 %p1, %r11, %s_cols;
            @%p1 bra F16_DONE;
            cvt.f32.f16 %f2, %hh;
            shr.u32 %r12, %r5, 16;
            NAN_HIGH
            st.global.f32 [%rd1+4], %f2;
            F16_DONE:
            }
            """.Replace("NAN_LOW", KeepNaN("%f1"), StringComparison.Ordinal).Replace("NAN_HIGH", KeepNaN("%f2"), StringComparison.Ordinal));

        // y[i] = scale[i % cols] · Σ_s part[s, i] over the k splits of int8_gemv_f32, in split order.
        Elementwise(sb, "int8_gemv_finish_f32", ["part", "s", "y"], [("u32", "cols"), ("u32", "size"), ("u32", "splits")], $"""
            mov.f32 %f1, {Zero};
            mov.u64 %rd2, %a_part;
            mul.wide.u32 %rd3, %s_size, 4;
            mov.u32 %r5, 0;
            FS:
            setp.ge.u32 %p1, %r5, %s_splits;
            @%p1 bra FS_END;
            ld.global.f32 %f2, [%rd2];
            add.f32 %f1, %f1, %f2;
            add.u64 %rd2, %rd2, %rd3;
            add.u32 %r5, %r5, 1;
            bra FS;
            FS_END:
            rem.u32 %r6, %i, %s_cols;
            mul.wide.u32 %rd4, %r6, 4;
            add.u64 %rd4, %rd4, %b_s;
            ld.global.f32 %f3, [%rd4];
            mul.f32 %f1, %f1, %f3;
            st.global.f32 [%a_y], %f1;
            """);

        // y[r, 4c..4c+3] = scale · Σ_k x[r, k] · q[k, 4c..4c+3]: one thread per (row, word). Adjacent threads read adjacent
        // words of a weight row, so a warp reads 128 contiguous bytes per step (the weights are read once per row).
        Elementwise(sb, "int8_matmul_f32", ["x", "q", "s", "y"], [("u32", "k"), ("u32", "n4"), ("u32", "cols")], $"""
            div.u32 %r5, %i, %s_n4;
            rem.u32 %r6, %i, %s_n4;
            mul.lo.u32 %r7, %r5, %s_k;
            mul.wide.u32 %rd1, %r7, 4;
            add.u64 %rd2, %b_x, %rd1;
            mul.wide.u32 %rd3, %r6, 4;
            add.u64 %rd4, %b_q, %rd3;
            mul.wide.u32 %rd5, %s_n4, 4;
            mov.f32 %f1, {Zero};
            mov.f32 %f2, {Zero};
            mov.f32 %f3, {Zero};
            mov.f32 %f4, {Zero};
            mov.u32 %r8, 0;
            QM_LOOP:
            setp.ge.u32 %p1, %r8, %s_k;
            @%p1 bra QM_END;
            ld.global.f32 %f5, [%rd2];
            ld.global.u32 %r9, [%rd4];
            bfe.s32 %r10, %r9, 0, 8;
            cvt.rn.f32.s32 %f6, %r10;
            fma.rn.f32 %f1, %f5, %f6, %f1;
            bfe.s32 %r10, %r9, 8, 8;
            cvt.rn.f32.s32 %f6, %r10;
            fma.rn.f32 %f2, %f5, %f6, %f2;
            bfe.s32 %r10, %r9, 16, 8;
            cvt.rn.f32.s32 %f6, %r10;
            fma.rn.f32 %f3, %f5, %f6, %f3;
            bfe.s32 %r10, %r9, 24, 8;
            cvt.rn.f32.s32 %f6, %r10;
            fma.rn.f32 %f4, %f5, %f6, %f4;
            add.u64 %rd2, %rd2, 4;
            add.u64 %rd4, %rd4, %rd5;
            add.u32 %r8, %r8, 1;
            bra QM_LOOP;
            QM_END:
            shl.b32 %r11, %r6, 2;
            mul.wide.u32 %rd6, %r11, 4;
            add.u64 %rd7, %b_s, %rd6;
            mul.lo.u32 %r12, %r5, %s_cols;
            add.u32 %r12, %r12, %r11;
            mul.wide.u32 %rd8, %r12, 4;
            add.u64 %rd9, %b_y, %rd8;
            """ + "\n" + StoreScaledColumns("QM", "%f1", "%f2", "%f3", "%f4"));

        // w[kk, 4c..4c+3] = q[kk, 4c..4c+3] · scale: one thread per packed word.
        Elementwise(sb, "int8_dequant_f32", ["q", "s", "w"], [("u32", "n4"), ("u32", "cols")], """
            ld.global.u32 %r9, [%a_q];
            div.u32 %r5, %i, %s_n4;
            rem.u32 %r6, %i, %s_n4;
            bfe.s32 %r10, %r9, 0, 8;
            cvt.rn.f32.s32 %f1, %r10;
            bfe.s32 %r10, %r9, 8, 8;
            cvt.rn.f32.s32 %f2, %r10;
            bfe.s32 %r10, %r9, 16, 8;
            cvt.rn.f32.s32 %f3, %r10;
            bfe.s32 %r10, %r9, 24, 8;
            cvt.rn.f32.s32 %f4, %r10;
            shl.b32 %r11, %r6, 2;
            mul.wide.u32 %rd6, %r11, 4;
            add.u64 %rd7, %b_s, %rd6;
            mul.lo.u32 %r12, %r5, %s_cols;
            add.u32 %r12, %r12, %r11;
            mul.wide.u32 %rd8, %r12, 4;
            add.u64 %rd9, %b_w, %rd8;
            """ + "\n" + StoreScaledColumns("QD", "%f1", "%f2", "%f3", "%f4"));

        // Int8 KV cache. Each cached row (one head, one position) holds dim values as bytes packed four per word
        // (words = ceil(dim / 4)) and one scale (max |x| / 127). The position comes from the device.

        // Quantizes source rows [heads·steps, dim] into slots [head, position + step]: one thread per row.
        Elementwise(sb, "kv_write_int8", ["src", "cache", "scales", "position"], [("u32", "steps"), ("u32", "capacity"), ("u32", "dim"), ("u32", "words")], $$"""
            div.u32 %r5, %i, %s_steps;
            rem.u32 %r6, %i, %s_steps;
            ld.global.f32 %f1, [%b_position];
            cvt.rzi.u32.f32 %r7, %f1;
            add.u32 %r7, %r7, %r6;
            mul.lo.u32 %r8, %i, %s_dim;
            mul.wide.u32 %rd1, %r8, 4;
            add.u64 %rd2, %b_src, %rd1;
            mov.f32 %f2, {{Zero}};
            mov.u32 %r9, 0;
            KW_MAX:
            setp.ge.u32 %p1, %r9, %s_dim;
            @%p1 bra KW_MAXED;
            mul.wide.u32 %rd3, %r9, 4;
            add.u64 %rd4, %rd2, %rd3;
            ld.global.f32 %f3, [%rd4];
            abs.f32 %f3, %f3;
            max.f32 %f2, %f2, %f3;
            add.u32 %r9, %r9, 1;
            bra KW_MAX;
            KW_MAXED:
            div.rn.f32 %f4, %f2, 0f42FE0000;
            setp.gt.f32 %p2, %f2, {{Zero}};
            mov.f32 %f5, {{Zero}};
            @%p2 rcp.rn.f32 %f5, %f4;
            mul.lo.u32 %r10, %r5, %s_capacity;
            add.u32 %r10, %r10, %r7;
            mul.wide.u32 %rd5, %r10, 4;
            add.u64 %rd6, %b_scales, %rd5;
            st.global.f32 [%rd6], %f4;
            mul.lo.u32 %r11, %r10, %s_words;
            mul.wide.u32 %rd7, %r11, 4;
            add.u64 %rd8, %b_cache, %rd7;
            mov.u32 %r12, 0;
            KW_WORD:
            setp.ge.u32 %p3, %r12, %s_words;
            @%p3 bra KW_DONE;
            mov.u32 %r13, 0;
            """ + "\n" + string.Concat(Enumerable.Range(0, 4).Select(t => string.Create(CultureInfo.InvariantCulture, $"""
            shl.b32 %r14, %r12, 2;
            add.u32 %r14, %r14, {t};
            setp.ge.u32 %p4, %r14, %s_dim;
            mov.f32 %f6, {Zero};
            mul.wide.u32 %rd9, %r14, 4;
            add.u64 %rd10, %rd2, %rd9;
            @!%p4 ld.global.f32 %f6, [%rd10];
            mul.f32 %f7, %f6, %f5;
            cvt.rni.s32.f32 %r15, %f7;
            min.s32 %r15, %r15, 127;
            max.s32 %r15, %r15, -127;
            and.b32 %r15, %r15, 255;
            shl.b32 %r15, %r15, {8 * t};
            or.b32 %r13, %r13, %r15;

            """))) + """
            mul.wide.u32 %rd11, %r12, 4;
            add.u64 %rd12, %rd8, %rd11;
            st.global.u32 [%rd12], %r13;
            add.u32 %r12, %r12, 1;
            bra KW_WORD;
            KW_DONE:
            """);

        // scores[r, t, c] = scale[r, c] · Σ_d q[r, t, d] · k[r, c, d]: one thread per (r, t, c).
        Elementwise(sb, "attn_scores_int8", ["q", "cache", "scales", "y"], [("u32", "steps"), ("u32", "capacity"), ("u32", "dim"), ("u32", "words")], $$"""
            rem.u32 %r5, %i, %s_capacity;
            div.u32 %r6, %i, %s_capacity;
            div.u32 %r7, %r6, %s_steps;
            mul.lo.u32 %r8, %r6, %s_dim;
            mul.wide.u32 %rd1, %r8, 4;
            add.u64 %rd2, %b_q, %rd1;
            mul.lo.u32 %r9, %r7, %s_capacity;
            add.u32 %r9, %r9, %r5;
            mul.lo.u32 %r10, %r9, %s_words;
            mul.wide.u32 %rd3, %r10, 4;
            add.u64 %rd4, %b_cache, %rd3;
            mov.f32 %f1, {{Zero}};
            mov.u32 %r11, 0;
            AS_LOOP:
            setp.ge.u32 %p1, %r11, %s_words;
            @%p1 bra AS_END;
            ld.global.u32 %r12, [%rd4];
            """ + "\n" + string.Concat(Enumerable.Range(0, 4).Select(t => string.Create(CultureInfo.InvariantCulture, $"""
            shl.b32 %r13, %r11, 2;
            add.u32 %r13, %r13, {t};
            setp.ge.u32 %p2, %r13, %s_dim;
            @%p2 bra AS_NEXT;
            mul.wide.u32 %rd5, %r13, 4;
            add.u64 %rd6, %rd2, %rd5;
            ld.global.f32 %f2, [%rd6];
            bfe.s32 %r14, %r12, {8 * t}, 8;
            cvt.rn.f32.s32 %f3, %r14;
            fma.rn.f32 %f1, %f2, %f3, %f1;

            """))) + """
            AS_NEXT:
            add.u64 %rd4, %rd4, 4;
            add.u32 %r11, %r11, 1;
            bra AS_LOOP;
            AS_END:
            mul.wide.u32 %rd7, %r9, 4;
            add.u64 %rd8, %b_scales, %rd7;
            ld.global.f32 %f4, [%rd8];
            mul.f32 %f1, %f1, %f4;
            st.global.f32 [%a_y], %f1;
            """);

        // out[r, t, 4w..4w+3] = Σ_c weights[r, t, c] · scale[r, c] · v[r, c, 4w..4w+3]: one thread per (r, t, word).
        Elementwise(sb, "attn_context_int8", ["w", "cache", "scales", "y"], [("u32", "steps"), ("u32", "capacity"), ("u32", "dim"), ("u32", "words")], $$"""
            rem.u32 %r5, %i, %s_words;
            div.u32 %r6, %i, %s_words;
            div.u32 %r7, %r6, %s_steps;
            mul.lo.u32 %r8, %r6, %s_capacity;
            mul.wide.u32 %rd1, %r8, 4;
            add.u64 %rd2, %b_w, %rd1;
            mul.lo.u32 %r9, %r7, %s_capacity;
            mul.wide.u32 %rd3, %r9, 4;
            add.u64 %rd4, %b_scales, %rd3;
            mul.lo.u32 %r10, %r9, %s_words;
            add.u32 %r10, %r10, %r5;
            mul.wide.u32 %rd5, %r10, 4;
            add.u64 %rd6, %b_cache, %rd5;
            mul.wide.u32 %rd7, %s_words, 4;
            mov.f32 %f1, {{Zero}};
            mov.f32 %f2, {{Zero}};
            mov.f32 %f3, {{Zero}};
            mov.f32 %f4, {{Zero}};
            mov.u32 %r11, 0;
            AC_LOOP:
            setp.ge.u32 %p1, %r11, %s_capacity;
            @%p1 bra AC_END;
            ld.global.f32 %f5, [%rd2];
            ld.global.f32 %f6, [%rd4];
            mul.f32 %f5, %f5, %f6;
            ld.global.u32 %r12, [%rd6];
            bfe.s32 %r13, %r12, 0, 8;
            cvt.rn.f32.s32 %f7, %r13;
            fma.rn.f32 %f1, %f5, %f7, %f1;
            bfe.s32 %r13, %r12, 8, 8;
            cvt.rn.f32.s32 %f7, %r13;
            fma.rn.f32 %f2, %f5, %f7, %f2;
            bfe.s32 %r13, %r12, 16, 8;
            cvt.rn.f32.s32 %f7, %r13;
            fma.rn.f32 %f3, %f5, %f7, %f3;
            bfe.s32 %r13, %r12, 24, 8;
            cvt.rn.f32.s32 %f7, %r13;
            fma.rn.f32 %f4, %f5, %f7, %f4;
            add.u64 %rd2, %rd2, 4;
            add.u64 %rd4, %rd4, 4;
            add.u64 %rd6, %rd6, %rd7;
            add.u32 %r11, %r11, 1;
            bra AC_LOOP;
            AC_END:
            shl.b32 %r14, %r5, 2;
            mul.lo.u32 %r15, %r6, %s_dim;
            add.u32 %r15, %r15, %r14;
            mul.wide.u32 %rd8, %r15, 4;
            add.u64 %rd9, %b_y, %rd8;
            """ + "\n" + string.Concat(Enumerable.Range(0, 4).Select(t => string.Create(CultureInfo.InvariantCulture, $"""
            add.u32 %r13, %r14, {t};
            setp.ge.u32 %p2, %r13, %s_dim;
            @%p2 bra AC_STORED;
            st.global.f32 [%rd9+{4 * t}], %f{t + 1};

            """))) + "AC_STORED:");
    }

    // Stores value[t] · scale[column + t] for the (up to) four columns starting at %r11 that are below n.
    // Expects %rd7 = &scale[column] and %rd9 = &out[row, column].
    private static string StoreScaledColumns(string prefix, params string[] values)
    {
        var sb = new StringBuilder();
        for (int t = 0; t < values.Length; t++)
        {
            sb.AppendLine(CultureInfo.InvariantCulture, $"add.u32 %r13, %r11, {t};");
            sb.AppendLine("setp.ge.u32 %p2, %r13, %s_cols;");
            sb.AppendLine($"@%p2 bra {prefix}_STORED;");
            sb.AppendLine(CultureInfo.InvariantCulture, $"ld.global.f32 %f7, [%rd7+{4 * t}];");
            sb.AppendLine($"mul.f32 %f8, {values[t]}, %f7;");
            sb.AppendLine(CultureInfo.InvariantCulture, $"st.global.f32 [%rd9+{4 * t}], %f8;");
        }

        sb.Append($"{prefix}_STORED:");
        return sb.ToString();
    }

    // y[r, j] = scale[j] · Σ_k x[r, k] · q[k, j] for m ≤ 8 rows, reading each packed weight word once. Block: 32 lanes
    // (a word = 4 columns each, so a warp reads 128 contiguous bytes) × 16 slices of k, four k per slice in flight;
    // grid: x = ⌈words / 32⌉, y = k splits of `chunk` rows each (enough blocks to fill the GPU when the matrix is
    // narrow). Slices are added in shared memory; with one split the result is scaled and stored, otherwise each
    // split's partial sums go to part[split, r, j] and the last block of the column range adds them in split order.
    // bf16: bf16_gemv_f32, the same for bfloat16 weights packed two per word (low half = the even column), no scales.
    // int4: int4_gemv_f32, signed nibbles packed eight per word (nibble c = column 8w + c) with a scale per group of
    // Int4Group rows and column, s[g, j] (rows of 8·words scales); a slice takes four consecutive rows (one group).
    //
    // multi (…_gemv_multi_f32): up to three products sharing the input x (query/key/value, gate/up), grid z selecting the
    // product: product j has its own weights q_j, scales s_j, output y_j [m, n_j] and optional bias b_j (0: none); blocks
    // past a product's columns return at once, and split partials and arrival counters get a region per product.
    //
    // gated 1 (…_gemv_silu_f32) / 2 (…_gemv_gelu_f32): the input is act(x) · up, computed as it is read (the gated
    // feed-forward's down projection without a separate activation pass); p_up follows p_counters.
    //
    // addNorm (…_gemv_addnorm_f32): then sum = res + y and norm = RMS-normalized sum · (gain + offset) per row (a
    // residual addition and the next normalization): the last block to finish its column range (counted in
    // counters[gridDim.x], reset afterwards) computes them over whole rows. p_res, p_sum, p_gain, p_norm, p_eps and
    // p_offset follow p_counters.
    //
    // normAdd (…_gemv_normaddnorm_f32, with addNorm): y's own RMS normalization · (ygain + yoffset) first, then sum = res
    // + that and norm as above (a sandwich block's projection, the norm after it, the residual addition and the next
    // norm). The last block reduces with the first RowThreads threads in the row kernels' order (rms_norm_affine_f32 and
    // add_rms_norm_affine_f32: each thread's columns tid, tid + RowThreads, … in turn, the warp's butterfly, then the
    // warps in order), multiplies and adds with explicit rounding (nothing contracted into a fused multiply-add), so its
    // three outputs equal the product followed by those two kernels bit for bit. p_ygain, p_yeps and p_yoffset follow
    // p_offset.
    //
    // gatedPair (…_gemv_multi_act_f32): the multi kernel for the gate (product 0) and up (product 1) projections of
    // equal widths, then hidden = act(gate) · up (p_act as gated_act_f32's kind): the second product to finish a column
    // range (counted in counters[2·cstride + column block], reset afterwards) computes it for those columns. p_act and
    // p_hidden follow p_n2.
    //
    // rows (…_gemv_r1_f32 / _r2 / _r4, the plain kernels only): compiled for at most that many rows of x instead of
    // GemvRows. Each output takes the eight-row kernel's operations in its order (the same rows of k per slice, the same
    // multiply-adds, the slices and splits added alike), so the results are equal bit for bit; only the accumulators
    // and the predicated-off rows' loads and multiply-adds are gone. The eight-row kernel issued all eight rows'
    // instructions for one row and kept 8 · cpw accumulators live (64 for int4), so a multiprocessor held one block of
    // it: one row reached about half the memory bandwidth (int8) or less (int4) that bfloat16 did on the same card.
    // Deep (rows · cpw ≤ 8: int8 to 2 rows, int4 1 row, bf16 to 4 rows): a pass loads eight words per thread before
    // using any (int8 / bf16: rows kk, kk + 16, …, kk + 112; int4: two 64-row blocks, each with its group's scales), the
    // shorter passes after it, so twice the bytes are in flight; one row also loads each step's input into a register
    // of its own (rule 80: all eight loads issue together).
    private static void Int8Gemv(StringBuilder sb, bool bf16 = false, bool int4 = false, bool multi = false, int gated = 0, bool addNorm = false,
        bool gatedPair = false, bool normAdd = false, int rows = GemvRows)
    {
        if (rows != GemvRows && (multi || gated != 0 || addNorm || !GemvRowVariants.Contains(rows)))
        {
            throw new ArgumentException($"The few-row kernels compiled for {rows} rows are the plain ones, for the counts of GemvRowVariants.", nameof(rows));
        }

        int cpw = bf16 ? 2 : int4 ? 8 : 4, columns = 32 * cpw;
        bool deep = rows != GemvRows && rows * cpw <= 8, ownInputs = deep && rows == 1;
        int acc = rows * cpw, dec = Math.Max(32, acc), sc = dec + 9, red = dec + (deep && int4 ? 28 : 20);
        string part = bf16 ? "h_part" : int4 ? "i4_part" : "i8_part";
        string name = (bf16 ? "bf16_gemv" : int4 ? "int4_gemv" : "int8_gemv") + (rows != GemvRows ? string.Create(CultureInfo.InvariantCulture, $"_r{rows}") : "")
            + (multi ? (gatedPair ? "_multi_act_f32" : "_multi_f32") : gated == 1 ? "_silu_f32" : gated == 2 ? "_gelu_f32" : normAdd ? "_normaddnorm_f32" : addNorm ? "_addnorm_f32" : "_f32");
        string finished = addNorm ? "COLS_DONE" : gatedPair ? "PAIR" : "DONE";
        int gt = red + 9;
        bool scaled = !bf16 && !int4;
        var s = new StringBuilder();
        string parameters = multi
            ? """
                .param .u64 p_x, .param .u64 p_part, .param .u32 p_m, .param .u32 p_k, .param .u32 p_chunk, .param .u32 p_splits,
                .param .u64 p_counters, .param .u32 p_cstride, .param .u32 p_nmax,
                .param .u64 p_q0, .param .u64 p_s0, .param .u64 p_y0, .param .u64 p_b0, .param .u32 p_n0,
                .param .u64 p_q1, .param .u64 p_s1, .param .u64 p_y1, .param .u64 p_b1, .param .u32 p_n1,
                .param .u64 p_q2, .param .u64 p_s2, .param .u64 p_y2, .param .u64 p_b2, .param .u32 p_n2
              """ + (gatedPair ? ", .param .u32 p_act, .param .u64 p_hidden" : "")
            : """
                .param .u64 p_x, .param .u64 p_q, .param .u64 p_s, .param .u64 p_y, .param .u64 p_part,
                .param .u32 p_m, .param .u32 p_n, .param .u32 p_k, .param .u32 p_n4, .param .u32 p_chunk, .param .u32 p_splits,
                .param .u64 p_counters
              """ + (gated > 0 ? ", .param .u64 p_up" : "")
              + (addNorm ? ", .param .u64 p_res, .param .u64 p_sum, .param .u64 p_gain, .param .u64 p_norm, .param .f32 p_eps, .param .f32 p_offset" : "")
              + (normAdd ? ", .param .u64 p_ygain, .param .f32 p_yeps, .param .f32 p_yoffset" : "");
        string loads = multi
            ? string.Create(CultureInfo.InvariantCulture, $$"""
                mov.u32 %r39, %ctaid.z;
                setp.eq.u32 %p21, %r39, 1;
                setp.eq.u32 %p22, %r39, 2;
                ld.param.u64 %rd1, [p_x];
                ld.param.u64 %rd2, [p_q0];
                @%p21 ld.param.u64 %rd2, [p_q1];
                @%p22 ld.param.u64 %rd2, [p_q2];
                ld.param.u64 %rd3, [p_s0];
                @%p21 ld.param.u64 %rd3, [p_s1];
                @%p22 ld.param.u64 %rd3, [p_s2];
                ld.param.u64 %rd4, [p_y0];
                @%p21 ld.param.u64 %rd4, [p_y1];
                @%p22 ld.param.u64 %rd4, [p_y2];
                ld.param.u64 %rd25, [p_b0];
                @%p21 ld.param.u64 %rd25, [p_b1];
                @%p22 ld.param.u64 %rd25, [p_b2];
                ld.param.u32 %r2, [p_n0];
                @%p21 ld.param.u32 %r2, [p_n1];
                @%p22 ld.param.u32 %r2, [p_n2];
                ld.param.u64 %rd5, [p_part];
                cvta.to.global.u64 %rd1, %rd1;
                cvta.to.global.u64 %rd2, %rd2;
                cvta.to.global.u64 %rd3, %rd3;
                cvta.to.global.u64 %rd4, %rd4;
                cvta.to.global.u64 %rd5, %rd5;
                setp.ne.u64 %p23, %rd25, 0;
                cvta.to.global.u64 %rd25, %rd25;
                ld.param.u32 %r1, [p_m];
                ld.param.u32 %r3, [p_k];
                add.u32 %r4, %r2, {{cpw - 1}};
                shr.u32 %r4, %r4, {{(int)Math.Log2(cpw)}};
                ld.param.u32 %r30, [p_chunk];
                ld.param.u32 %r31, [p_splits];
                mov.u32 %r36, %ctaid.x;
                shl.b32 %r36, %r36, 5;
                setp.ge.u32 %p21, %r36, %r4;
                @%p21 ret;
                ld.param.u32 %r37, [p_nmax];
                mul.lo.u32 %r37, %r37, %r1;
                mul.lo.u32 %r37, %r37, %r31;
                mul.wide.u32 %rd26, %r37, %r39;
                shl.b64 %rd26, %rd26, 2;
                add.u64 %rd5, %rd5, %rd26;
              """)
            : """
                ld.param.u64 %rd1, [p_x];
                ld.param.u64 %rd2, [p_q];
                ld.param.u64 %rd3, [p_s];
                ld.param.u64 %rd4, [p_y];
                ld.param.u64 %rd5, [p_part];
                cvta.to.global.u64 %rd1, %rd1;
                cvta.to.global.u64 %rd2, %rd2;
                cvta.to.global.u64 %rd3, %rd3;
                cvta.to.global.u64 %rd4, %rd4;
                cvta.to.global.u64 %rd5, %rd5;
                ld.param.u32 %r1, [p_m];
                ld.param.u32 %r2, [p_n];
                ld.param.u32 %r3, [p_k];
                ld.param.u32 %r4, [p_n4];
                ld.param.u32 %r30, [p_chunk];
                ld.param.u32 %r31, [p_splits];
              """ + (gated > 0 ? "\n    ld.param.u64 %rd29, [p_up];\n    cvta.to.global.u64 %rd29, %rd29;\n    sub.u64 %rd29, %rd29, %rd1;" : "");
        // Written for blocks of Int8GemvThreads (16 slices of k × 32 lanes), declared so ptxas keeps its registers within
        // what such a block may use; %x (and %u, the up projection of a gated kernel): each row's input of a k step. The
        // normaddnorm kernels also state one block per multiprocessor: with their last block's loops ptxas otherwise aimed at
        // two (64 registers on sm_80) and spilled.
        s.AppendLine(CultureInfo.InvariantCulture, $$"""
            .visible .entry {{name}}(
            {{parameters}}
            )
            .maxntid {{Int8GemvThreads}}, 1, 1{{(normAdd ? "\n.minnctapersm 1" : "")}}
            {
                .reg .pred %p<24>;
                .reg .f32 %f<{{red + (normAdd ? 26 : 12)}}>;
                .reg .f32 %x<{{(ownInputs ? 8 : rows)}}>;{{(gated > 0 ? string.Create(CultureInfo.InvariantCulture, $"\n    .reg .f32 %u<{GemvRows}>;") : "")}}
                .reg .b32 %r<{{(deep ? 48 : 40)}}>;
                .reg .b64 %rd<{{(deep ? 36 : 32)}}>;
                .shared .align 4 .f32 {{part}}[{{16 * columns}}];
            {{loads}}
                mov.u32 %r5, %tid.x;
                and.b32 %r6, %r5, 31;
                shr.u32 %r7, %r5, 5;
                mov.u32 %r8, %ctaid.x;
                shl.b32 %r8, %r8, 5;
                add.u32 %r8, %r8, %r6;
                setp.lt.u32 %p20, %r8, %r4;
                mov.u32 %r9, %ctaid.y;
                mul.lo.u32 %r10, %r9, %r30;
                add.u32 %r11, %r10, %r30;
                min.u32 %r11, %r11, %r3;
                mul.wide.u32 %rd6, %r4, 64;
                mul.wide.u32 %rd7, %r3, 4;
                mul.wide.u32 %rd8, %r4, 4;
                shl.b32 %r36, %r4, 3;
            """);
        for (int r = 0; r < rows; r++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    setp.lt.u32 %p{r}, {r}, %r1;");
            for (int c = 0; c < cpw; c++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    mov.f32 %f{r * cpw + c}, 0f00000000;");
            }
        }

        if (int4)
        {
            for (int c = 0; c < (deep ? 16 : 8); c++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    mov.f32 %f{sc + c}, 0f00000000;");
            }
        }

        // One k step: word %r{word} holds cpw columns; x[r, kk] is at %rd11 + {offset} + r·k·4. int4: the group's scales
        // are %f{scales} … %f{scales + 7}; row r's input goes to %x{input + r}.
        string Step(string wordReg, string xOffset, int scales = -1, int input = 0)
        {
            scales = scales < 0 ? sc : scales;
            var t = new StringBuilder();
            if (bf16)
            {
                // bfloat16 is the top half of a float32: shift the even column up, mask the odd one.
                t.AppendLine($"shl.b32 %r20, {wordReg}, 16;");
                t.AppendLine(CultureInfo.InvariantCulture, $"mov.b32 %f{dec}, %r20;");
                t.AppendLine($"and.b32 %r20, {wordReg}, 0xFFFF0000;");
                t.AppendLine(CultureInfo.InvariantCulture, $"mov.b32 %f{dec + 1}, %r20;");
            }
            else
            {
                for (int c = 0; c < cpw; c++)
                {
                    int bits = int4 ? 4 : 8;
                    t.AppendLine(CultureInfo.InvariantCulture, $"bfe.s32 %r20, {wordReg}, {bits * c}, {bits};");
                    t.AppendLine(CultureInfo.InvariantCulture, $"cvt.rn.f32.s32 %f{dec + c}, %r20;");
                    if (int4)
                    {
                        t.AppendLine(CultureInfo.InvariantCulture, $"mul.f32 %f{dec + c}, %f{dec + c}, %f{scales + c};");
                    }
                }
            }

            // Every row's input in a register of its own (%x, and %u for a gated kernel's up projection), all loaded before
            // any is used, so the step's loads are in flight together (one register shared by the rows and written under
            // their predicates made each load wait for the previous row's multiply-adds). Each row's sums take the same
            // operations in the same order.
            t.AppendLine("mov.u64 %rd12, %rd11;");
            for (int r = 0; r < rows; r++)
            {
                t.AppendLine(CultureInfo.InvariantCulture, $"@%p{r} ld.global.f32 %x{input + r}, [%rd12+{xOffset}];");
                if (gated > 0)
                {
                    t.AppendLine("add.u64 %rd28, %rd12, %rd29;");
                    t.AppendLine(CultureInfo.InvariantCulture, $"@%p{r} ld.global.f32 %u{r}, [%rd28+{xOffset}];");
                }

                t.AppendLine("add.u64 %rd12, %rd12, %rd7;");
            }

            for (int r = 0; r < rows; r++)
            {
                if (gated > 0)
                {
                    // x = act(gate) · up, with the forward formulas of gated_act_f32.
                    t.AppendLine(gated == 1
                        ? string.Create(CultureInfo.InvariantCulture, $"""
                          mul.f32 %f{gt}, %x{r}, {F(-1.4426950408889634f)};
                          ex2.approx.ftz.f32 %f{gt}, %f{gt};
                          add.f32 %f{gt}, %f{gt}, {One};
                          rcp.rn.f32 %f{gt}, %f{gt};
                          mul.f32 %x{r}, %x{r}, %f{gt};
                          """)
                        : string.Create(CultureInfo.InvariantCulture, $"""
                          mul.f32 %f{gt}, %x{r}, %x{r};
                          mul.f32 %f{gt}, %f{gt}, %x{r};
                          fma.rn.f32 %f{gt}, %f{gt}, {F(0.044715f)}, %x{r};
                          mul.f32 %f{gt}, %f{gt}, {F(0.7978845608f)};
                          mul.f32 %f{gt}, %f{gt}, {F(2.8853900817779268f)};
                          ex2.approx.ftz.f32 %f{gt}, %f{gt};
                          add.f32 %f{gt}, %f{gt}, {One};
                          rcp.rn.f32 %f{gt}, %f{gt};
                          fma.rn.f32 %f{gt}, %f{gt}, {F(-2f)}, {One};
                          add.f32 %f{gt}, %f{gt}, {One};
                          mul.f32 %f{gt}, %f{gt}, %x{r};
                          mul.f32 %x{r}, %f{gt}, {F(0.5f)};
                          """));
                    t.AppendLine(CultureInfo.InvariantCulture, $"mul.f32 %x{r}, %x{r}, %u{r};");
                }

                for (int c = 0; c < cpw; c++)
                {
                    t.AppendLine(CultureInfo.InvariantCulture, $"@%p{r} fma.rn.f32 %f{r * cpw + c}, %x{input + r}, %f{dec + c}, %f{r * cpw + c};");
                }
            }

            return t.ToString();
        }

        // Int4: the scales of row %r{row}'s group for this thread's eight columns (two 16-byte loads) into %f{scales} …
        // %f{scales + 7}.
        string GroupScales(string row, int scales = -1)
        {
            scales = scales < 0 ? sc : scales;
            return string.Create(CultureInfo.InvariantCulture, $"""
                    shr.u32 %r37, {row}, {(int)Math.Log2(Int4Group)};
                    mul.wide.u32 %rd15, %r37, %r36;
                    cvt.u64.u32 %rd16, %r8;
                    shl.b64 %rd16, %rd16, 3;
                    add.u64 %rd15, %rd15, %rd16;
                    shl.b64 %rd15, %rd15, 2;
                    add.u64 %rd15, %rd15, %rd3;
                    @%p20 ld.global.v4.f32 {"{"}%f{scales}, %f{scales + 1}, %f{scales + 2}, %f{scales + 3}{"}"}, [%rd15];
                    @%p20 ld.global.v4.f32 {"{"}%f{scales + 4}, %f{scales + 5}, %f{scales + 6}, %f{scales + 7}{"}"}, [%rd15+16];
                """);
        }

        // Deep kernels: the eight words of a pass in %r21 … %r24, %r40 … %r43, loaded before any is used (the second four
        // from %rd32 on, `second` bytes past the first word), each %rd9 += `stride` after the one before.
        string EightWords(string stride, string second) => string.Create(CultureInfo.InvariantCulture, $"""
                    {second}
                    mov.u32 %r21, 0;
                    mov.u32 %r22, 0;
                    mov.u32 %r23, 0;
                    mov.u32 %r24, 0;
                    mov.u32 %r40, 0;
                    mov.u32 %r41, 0;
                    mov.u32 %r42, 0;
                    mov.u32 %r43, 0;
                    @%p20 ld.global.u32 %r21, [%rd9];
                    add.u64 %rd9, %rd9, {stride};
                    @%p20 ld.global.u32 %r22, [%rd9];
                    add.u64 %rd9, %rd9, {stride};
                    @%p20 ld.global.u32 %r23, [%rd9];
                    add.u64 %rd9, %rd9, {stride};
                    @%p20 ld.global.u32 %r24, [%rd9];
                    @%p20 ld.global.u32 %r40, [%rd32];
                    add.u64 %rd32, %rd32, {stride};
                    @%p20 ld.global.u32 %r41, [%rd32];
                    add.u64 %rd32, %rd32, {stride};
                    @%p20 ld.global.u32 %r42, [%rd32];
                    add.u64 %rd32, %rd32, {stride};
                    @%p20 ld.global.u32 %r43, [%rd32];
                """);

        // The input register base of step u of a deep pass: one of its own per step for one row, else shared.
        int Input(int u) => ownInputs ? u : 0;

        if (int4)
        {
            // Slice t takes rows base + 4t .. base + 4t + 3 of each 64-row block (one group, so one set of scales).
            s.AppendLine("""
                    mov.u32 %r12, %r10;
                    shl.b32 %r34, %r7, 2;
                """);
            if (deep)
            {
                // Two 64-row blocks a pass while both fit, in block order (the order of the one-block passes), each
                // with its group's scales (%f{sc} … and %f{sc + 8} …); then the one-block passes.
                s.AppendLine("""
                        K8:
                            add.u32 %r13, %r12, 128;
                            setp.gt.u32 %p10, %r13, %r11;
                            @%p10 bra K4;
                            add.u32 %r35, %r12, %r34;
                            add.u32 %r44, %r35, 64;
                            mul.wide.u32 %rd9, %r35, %r4;
                            cvt.u64.u32 %rd10, %r8;
                            add.u64 %rd9, %rd9, %rd10;
                            shl.b64 %rd9, %rd9, 2;
                            add.u64 %rd9, %rd9, %rd2;
                        """);
                s.AppendLine(EightWords("%rd8", "shl.b64 %rd32, %rd6, 2;\n    add.u64 %rd32, %rd32, %rd9;"));
                s.AppendLine(GroupScales("%r35"));
                s.AppendLine(GroupScales("%r44", sc + 8));
                s.AppendLine("""
                            mul.wide.u32 %rd11, %r35, 4;
                            add.u64 %rd11, %rd11, %rd1;
                        """);
                s.AppendLine(Step("%r21", "0", sc, Input(0)));
                s.AppendLine(Step("%r22", "4", sc, Input(1)));
                s.AppendLine(Step("%r23", "8", sc, Input(2)));
                s.AppendLine(Step("%r24", "12", sc, Input(3)));
                s.AppendLine(Step("%r40", "256", sc + 8, Input(4)));
                s.AppendLine(Step("%r41", "260", sc + 8, Input(5)));
                s.AppendLine(Step("%r42", "264", sc + 8, Input(6)));
                s.AppendLine(Step("%r43", "268", sc + 8, Input(7)));
                s.AppendLine("""
                            add.u32 %r12, %r12, 128;
                            bra K8;
                        """);
            }

            s.AppendLine("""
                K4:
                    add.u32 %r13, %r12, 64;
                    setp.gt.u32 %p10, %r13, %r11;
                    @%p10 bra K1S;
                    add.u32 %r35, %r12, %r34;
                    mul.wide.u32 %rd9, %r35, %r4;
                    cvt.u64.u32 %rd10, %r8;
                    add.u64 %rd9, %rd9, %rd10;
                    shl.b64 %rd9, %rd9, 2;
                    add.u64 %rd9, %rd9, %rd2;
                    mov.u32 %r21, 0;
                    mov.u32 %r22, 0;
                    mov.u32 %r23, 0;
                    mov.u32 %r24, 0;
                    @%p20 ld.global.u32 %r21, [%rd9];
                    add.u64 %rd9, %rd9, %rd8;
                    @%p20 ld.global.u32 %r22, [%rd9];
                    add.u64 %rd9, %rd9, %rd8;
                    @%p20 ld.global.u32 %r23, [%rd9];
                    add.u64 %rd9, %rd9, %rd8;
                    @%p20 ld.global.u32 %r24, [%rd9];
                """);
            s.AppendLine(GroupScales("%r35"));
            s.AppendLine("""
                    mul.wide.u32 %rd11, %r35, 4;
                    add.u64 %rd11, %rd11, %rd1;
                """);
            s.AppendLine(Step("%r21", "0"));
            s.AppendLine(Step("%r22", "4"));
            s.AppendLine(Step("%r23", "8"));
            s.AppendLine(Step("%r24", "12"));
            s.AppendLine("""
                    add.u32 %r12, %r12, 64;
                    bra K4;
                K1S:
                    add.u32 %r12, %r12, %r7;
                """);
        }
        else
        {
            s.AppendLine("""
                    add.u32 %r12, %r10, %r7;
                """);
            if (deep)
            {
                // Rows kk, kk + 16, …, kk + 112 a pass while all eight are in the chunk (the four-row passes' order);
                // then the four-row passes.
                s.AppendLine("""
                        K8:
                            add.u32 %r13, %r12, 112;
                            setp.ge.u32 %p10, %r13, %r11;
                            @%p10 bra K4;
                            mul.wide.u32 %rd9, %r12, %r4;
                            cvt.u64.u32 %rd10, %r8;
                            add.u64 %rd9, %rd9, %rd10;
                            shl.b64 %rd9, %rd9, 2;
                            add.u64 %rd9, %rd9, %rd2;
                        """);
                s.AppendLine(EightWords("%rd6", "shl.b64 %rd32, %rd6, 2;\n    add.u64 %rd32, %rd32, %rd9;"));
                s.AppendLine("""
                            mul.wide.u32 %rd11, %r12, 4;
                            add.u64 %rd11, %rd11, %rd1;
                        """);
                s.AppendLine(Step("%r21", "0", input: Input(0)));
                s.AppendLine(Step("%r22", "64", input: Input(1)));
                s.AppendLine(Step("%r23", "128", input: Input(2)));
                s.AppendLine(Step("%r24", "192", input: Input(3)));
                s.AppendLine(Step("%r40", "256", input: Input(4)));
                s.AppendLine(Step("%r41", "320", input: Input(5)));
                s.AppendLine(Step("%r42", "384", input: Input(6)));
                s.AppendLine(Step("%r43", "448", input: Input(7)));
                s.AppendLine("""
                            add.u32 %r12, %r12, 128;
                            bra K8;
                        """);
            }

            s.AppendLine("""
                K4:
                    add.u32 %r13, %r12, 48;
                    setp.ge.u32 %p10, %r13, %r11;
                    @%p10 bra K1;
                    mul.wide.u32 %rd9, %r12, %r4;
                    cvt.u64.u32 %rd10, %r8;
                    add.u64 %rd9, %rd9, %rd10;
                    shl.b64 %rd9, %rd9, 2;
                    add.u64 %rd9, %rd9, %rd2;
                    mov.u32 %r21, 0;
                    mov.u32 %r22, 0;
                    mov.u32 %r23, 0;
                    mov.u32 %r24, 0;
                    @%p20 ld.global.u32 %r21, [%rd9];
                    add.u64 %rd9, %rd9, %rd6;
                    @%p20 ld.global.u32 %r22, [%rd9];
                    add.u64 %rd9, %rd9, %rd6;
                    @%p20 ld.global.u32 %r23, [%rd9];
                    add.u64 %rd9, %rd9, %rd6;
                    @%p20 ld.global.u32 %r24, [%rd9];
                    mul.wide.u32 %rd11, %r12, 4;
                    add.u64 %rd11, %rd11, %rd1;
                """);
            s.AppendLine(Step("%r21", "0"));
            s.AppendLine(Step("%r22", "64"));
            s.AppendLine(Step("%r23", "128"));
            s.AppendLine(Step("%r24", "192"));
            s.AppendLine("""
                    add.u32 %r12, %r12, 64;
                    bra K4;
                """);
        }

        s.AppendLine("""
            K1:
                setp.ge.u32 %p10, %r12, %r11;
                @%p10 bra KEND;
                mul.wide.u32 %rd9, %r12, %r4;
                cvt.u64.u32 %rd10, %r8;
                add.u64 %rd9, %rd9, %rd10;
                shl.b64 %rd9, %rd9, 2;
                add.u64 %rd9, %rd9, %rd2;
                mov.u32 %r21, 0;
                @%p20 ld.global.u32 %r21, [%rd9];
                mul.wide.u32 %rd11, %r12, 4;
                add.u64 %rd11, %rd11, %rd1;
            """);
        if (int4)
        {
            s.AppendLine(GroupScales("%r12"));
        }

        s.AppendLine(Step("%r21", "0"));
        s.AppendLine(CultureInfo.InvariantCulture, $$"""
                add.u32 %r12, %r12, 16;
                bra K1;
            KEND:
                mov.u32 %r14, {{part}};
                shl.b32 %r15, %r7, {{(int)Math.Log2(columns * 4)}};
                shl.b32 %r16, %r6, {{(int)Math.Log2(cpw * 4)}};
                add.u32 %r15, %r15, %r16;
                add.u32 %r15, %r15, %r14;
                shl.b32 %r17, %r5, 2;
                add.u32 %r17, %r17, %r14;
                mov.u32 %r18, %ctaid.x;
                shl.b32 %r18, %r18, {{(int)Math.Log2(columns)}};
                add.u32 %r18, %r18, %r5;
                setp.lt.u32 %p11, %r5, {{columns}};
                setp.lt.u32 %p12, %r18, %r2;
                and.pred %p11, %p11, %p12;
                setp.eq.u32 %p13, %r31, 1;
                mul.wide.u32 %rd13, %r18, 4;
                add.u64 %rd14, %rd13, %rd3;
            """);
        for (int r = 0; r < rows; r++)
        {
            // part[slice][lane·cpw + c] = acc[r][c]; then one thread per column adds the 16 slices of its column.
            s.AppendLine(CultureInfo.InvariantCulture, $"    @!%p{r} bra ROWS_DONE;");
            for (int c = 0; c < cpw; c++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    st.shared.f32 [%r15+{4 * c}], %f{r * cpw + c};");
            }

            s.AppendLine("    bar.sync 0;");
            s.AppendLine(CultureInfo.InvariantCulture, $"    @!%p11 bra ROW{r}_DONE;");
            s.AppendLine(CultureInfo.InvariantCulture, $"    mov.f32 %f{red}, 0f00000000;");
            for (int slice = 0; slice < 16; slice++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    ld.shared.f32 %f{red + 1}, [%r17+{slice * columns * 4}];");
                s.AppendLine(CultureInfo.InvariantCulture, $"    add.f32 %f{red}, %f{red}, %f{red + 1};");
            }

            s.AppendLine(CultureInfo.InvariantCulture, $"""
                    mad.lo.u32 %r25, %r9, %r1, {r};
                    mul.wide.u32 %rd15, %r25, %r2;
                    @%p13 bra ROW{r}_FINAL;
                    shl.b64 %rd16, %rd15, 2;
                    add.u64 %rd16, %rd16, %rd13;
                    add.u64 %rd16, %rd16, %rd5;
                    st.global.f32 [%rd16], %f{red};
                    bra ROW{r}_DONE;
                ROW{r}_FINAL:
                    {(scaled ? string.Create(CultureInfo.InvariantCulture, $"ld.global.f32 %f{red + 2}, [%rd14];\n    mul.f32 %f{red}, %f{red}, %f{red + 2};") : "")}
                    {(multi ? string.Create(CultureInfo.InvariantCulture, $"add.u64 %rd27, %rd25, %rd13;\n    @%p23 ld.global.f32 %f{red + 2}, [%rd27];\n    @%p23 add.f32 %f{red}, %f{red}, %f{red + 2};") : "")}
                    mul.wide.u32 %rd16, %r2, {4 * r};
                    add.u64 %rd16, %rd16, %rd13;
                    add.u64 %rd16, %rd16, %rd4;
                    st.global.f32 [%rd16], %f{red};
                ROW{r}_DONE:
                    bar.sync 0;
                """);
        }

        // With several splits, the last block of this column range to finish (counted per range, reset afterwards) adds
        // the splits' partial sums in split order and writes the scaled result: one launch, deterministic sums.
        s.AppendLine(CultureInfo.InvariantCulture, $$"""
            ROWS_DONE:
                @%p13 bra {{finished}};
                membar.gl;
                bar.sync 0;
                ld.param.u64 %rd17, [p_counters];
                cvta.to.global.u64 %rd17, %rd17;
                {{(multi ? "ld.param.u32 %r38, [p_cstride];\n    mul.lo.u32 %r38, %r38, %r39;\n    mul.wide.u32 %rd18, %r38, 4;\n    add.u64 %rd17, %rd17, %rd18;" : "")}}
                mov.u32 %r26, %ctaid.x;
                mul.wide.u32 %rd18, %r26, 4;
                add.u64 %rd17, %rd17, %rd18;
                setp.eq.u32 %p14, %r5, 0;
                mov.u32 %r27, 0;
                @%p14 atom.global.add.u32 %r27, [%rd17], 1;
                sub.u32 %r28, %r31, 1;
                setp.eq.u32 %p15, %r27, %r28;
                selp.u32 %r29, 1, 0, %p15;
                @%p14 st.shared.u32 [%r14], %r29;
                bar.sync 0;
                ld.shared.u32 %r29, [%r14];
                setp.eq.u32 %p15, %r29, 0;
                @%p15 bra DONE;
                membar.gl;
                @%p14 st.global.u32 [%rd17], 0;
                @!%p11 bra {{finished}};
                mul.wide.u32 %rd19, %r1, %r2;
                shl.b64 %rd19, %rd19, 2;
                mov.u32 %r32, 0;
            FIN_ROW:
                setp.ge.u32 %p16, %r32, %r1;
                @%p16 bra {{finished}};
                mul.wide.u32 %rd21, %r32, %r2;
                shl.b64 %rd21, %rd21, 2;
                add.u64 %rd21, %rd21, %rd13;
                add.u64 %rd22, %rd21, %rd5;
                mov.f32 %f{{red + 3}}, 0f00000000;
                mov.u32 %r33, 0;
            FIN_SPLIT:
                setp.ge.u32 %p17, %r33, %r31;
                @%p17 bra FIN_SPLIT_END;
                ld.global.cg.f32 %f{{red + 4}}, [%rd22];
                add.f32 %f{{red + 3}}, %f{{red + 3}}, %f{{red + 4}};
                add.u64 %rd22, %rd22, %rd19;
                add.u32 %r33, %r33, 1;
                bra FIN_SPLIT;
            FIN_SPLIT_END:
                {{(scaled ? string.Create(CultureInfo.InvariantCulture, $"ld.global.f32 %f{red + 5}, [%rd14];\n    mul.f32 %f{red + 3}, %f{red + 3}, %f{red + 5};") : "")}}
                {{(multi ? string.Create(CultureInfo.InvariantCulture, $"add.u64 %rd27, %rd25, %rd13;\n    @%p23 ld.global.f32 %f{red + 5}, [%rd27];\n    @%p23 add.f32 %f{red + 3}, %f{red + 3}, %f{red + 5};") : "")}}
                add.u64 %rd23, %rd21, %rd4;
                st.global.f32 [%rd23], %f{{red + 3}};
                add.u32 %r32, %r32, 1;
                bra FIN_ROW;
            """);
        if (addNorm)
        {
            // Every thread of every block gets here once its columns are final; the last block adds and normalizes.
            int t0 = red + 6, t1 = red + 7, t2 = red + 8, t3 = red + 9, t4 = red + 10, t5 = red + 11;
            s.AppendLine(CultureInfo.InvariantCulture, $$"""
                COLS_DONE:
                    membar.gl;
                    bar.sync 0;
                    ld.param.u64 %rd17, [p_counters];
                    cvta.to.global.u64 %rd17, %rd17;
                    mov.u32 %r26, %nctaid.x;
                    mul.wide.u32 %rd18, %r26, 4;
                    add.u64 %rd17, %rd17, %rd18;
                    setp.eq.u32 %p14, %r5, 0;
                    mov.u32 %r27, 0;
                    @%p14 atom.global.add.u32 %r27, [%rd17], 1;
                    sub.u32 %r28, %r26, 1;
                    setp.eq.u32 %p15, %r27, %r28;
                    selp.u32 %r29, 1, 0, %p15;
                    @%p14 st.shared.u32 [%r14], %r29;
                    bar.sync 0;
                    ld.shared.u32 %r29, [%r14];
                    setp.eq.u32 %p15, %r29, 0;
                    @%p15 bra DONE;
                    membar.gl;
                    @%p14 st.global.u32 [%rd17], 0;
                    bar.sync 0;
                """);
            if (normAdd)
            {
                s.AppendLine(NormAddNorm(red));
            }
            else
            {
                s.AppendLine(CultureInfo.InvariantCulture, $$"""
                        ld.param.u64 %rd18, [p_res];
                        ld.param.u64 %rd19, [p_sum];
                        ld.param.u64 %rd20, [p_gain];
                        ld.param.u64 %rd21, [p_norm];
                        cvta.to.global.u64 %rd18, %rd18;
                        cvta.to.global.u64 %rd19, %rd19;
                        cvta.to.global.u64 %rd20, %rd20;
                        cvta.to.global.u64 %rd21, %rd21;
                        ld.param.f32 %f{{t0}}, [p_eps];
                        ld.param.f32 %f{{t1}}, [p_offset];
                        cvt.rn.f32.u32 %f{{t2}}, %r2;
                        shl.b32 %r22, %r7, 2;
                        add.u32 %r22, %r22, %r14;
                        setp.eq.u32 %p18, %r6, 0;
                        mov.u32 %r32, 0;
                    AN_ROW:
                        setp.ge.u32 %p16, %r32, %r1;
                        @%p16 bra DONE;
                        mul.lo.u32 %r33, %r32, %r2;
                        mov.f32 %f{{t3}}, 0f00000000;
                        mov.u32 %r20, %r5;
                    AN_SUM:
                        setp.ge.u32 %p17, %r20, %r2;
                        @%p17 bra AN_SUM_END;
                        add.u32 %r21, %r33, %r20;
                        mul.wide.u32 %rd22, %r21, 4;
                        add.u64 %rd23, %rd22, %rd18;
                        ld.global.f32 %f{{t4}}, [%rd23];
                        add.u64 %rd23, %rd22, %rd4;
                        ld.global.cg.f32 %f{{t5}}, [%rd23];
                        add.f32 %f{{t4}}, %f{{t4}}, %f{{t5}};
                        add.u64 %rd23, %rd22, %rd19;
                        st.global.f32 [%rd23], %f{{t4}};
                        fma.rn.f32 %f{{t3}}, %f{{t4}}, %f{{t4}}, %f{{t3}};
                        add.u32 %r20, %r20, {{Int8GemvThreads}};
                        bra AN_SUM;
                    AN_SUM_END:
                        shfl.sync.bfly.b32 %f{{t5}}, %f{{t3}}, 16, 31, 0xffffffff;
                        add.f32 %f{{t3}}, %f{{t3}}, %f{{t5}};
                        shfl.sync.bfly.b32 %f{{t5}}, %f{{t3}}, 8, 31, 0xffffffff;
                        add.f32 %f{{t3}}, %f{{t3}}, %f{{t5}};
                        shfl.sync.bfly.b32 %f{{t5}}, %f{{t3}}, 4, 31, 0xffffffff;
                        add.f32 %f{{t3}}, %f{{t3}}, %f{{t5}};
                        shfl.sync.bfly.b32 %f{{t5}}, %f{{t3}}, 2, 31, 0xffffffff;
                        add.f32 %f{{t3}}, %f{{t3}}, %f{{t5}};
                        shfl.sync.bfly.b32 %f{{t5}}, %f{{t3}}, 1, 31, 0xffffffff;
                        add.f32 %f{{t3}}, %f{{t3}}, %f{{t5}};
                        @%p18 st.shared.f32 [%r22], %f{{t3}};
                        bar.sync 0;
                        mov.f32 %f{{t3}}, 0f00000000;
                    """);
                for (int w = 0; w < Int8GemvThreads / 32; w++)
                {
                    s.AppendLine(CultureInfo.InvariantCulture, $"    ld.shared.f32 %f{t5}, [%r14+{4 * w}];");
                    s.AppendLine(CultureInfo.InvariantCulture, $"    add.f32 %f{t3}, %f{t3}, %f{t5};");
                }

                s.AppendLine(CultureInfo.InvariantCulture, $$"""
                        bar.sync 0;
                        div.rn.f32 %f{{t3}}, %f{{t3}}, %f{{t2}};
                        add.f32 %f{{t3}}, %f{{t3}}, %f{{t0}};
                        sqrt.rn.f32 %f{{t3}}, %f{{t3}};
                        rcp.rn.f32 %f{{t3}}, %f{{t3}};
                        mov.u32 %r20, %r5;
                    AN_NORM:
                        setp.ge.u32 %p17, %r20, %r2;
                        @%p17 bra AN_NORM_END;
                        add.u32 %r21, %r33, %r20;
                        mul.wide.u32 %rd22, %r21, 4;
                        add.u64 %rd23, %rd22, %rd19;
                        ld.global.f32 %f{{t4}}, [%rd23];
                        mul.wide.u32 %rd24, %r20, 4;
                        add.u64 %rd24, %rd24, %rd20;
                        ld.global.f32 %f{{t5}}, [%rd24];
                        add.f32 %f{{t5}}, %f{{t5}}, %f{{t1}};
                        mul.f32 %f{{t4}}, %f{{t4}}, %f{{t3}};
                        mul.f32 %f{{t4}}, %f{{t4}}, %f{{t5}};
                        add.u64 %rd23, %rd22, %rd21;
                        st.global.f32 [%rd23], %f{{t4}};
                        add.u32 %r20, %r20, {{Int8GemvThreads}};
                        bra AN_NORM;
                    AN_NORM_END:
                        add.u32 %r32, %r32, 1;
                        bra AN_ROW;
                    """);
            }
        }

        if (gatedPair)
        {
            // Every thread of every block gets here once its columns of its product are final; the second of the two
            // products to finish a column range computes act(gate) · up for it (activation formulas of gated_act_f32).
            int g = red + 6, u = red + 7, a = red + 8, t1 = red + 9, t2 = red + 10, t3 = red + 11;
            s.AppendLine(CultureInfo.InvariantCulture, $$"""
                PAIR:
                    membar.gl;
                    bar.sync 0;
                    ld.param.u64 %rd17, [p_counters];
                    cvta.to.global.u64 %rd17, %rd17;
                    ld.param.u32 %r26, [p_cstride];
                    shl.b32 %r26, %r26, 1;
                    mov.u32 %r27, %ctaid.x;
                    add.u32 %r26, %r26, %r27;
                    mul.wide.u32 %rd18, %r26, 4;
                    add.u64 %rd17, %rd17, %rd18;
                    setp.eq.u32 %p14, %r5, 0;
                    mov.u32 %r27, 0;
                    @%p14 atom.global.add.u32 %r27, [%rd17], 1;
                    setp.eq.u32 %p15, %r27, 1;
                    selp.u32 %r29, 1, 0, %p15;
                    @%p14 st.shared.u32 [%r14], %r29;
                    bar.sync 0;
                    ld.shared.u32 %r29, [%r14];
                    setp.eq.u32 %p15, %r29, 0;
                    @%p15 bra DONE;
                    membar.gl;
                    @%p14 st.global.u32 [%rd17], 0;
                    setp.ge.u32 %p16, %r5, {{columns}};
                    @%p16 bra DONE;
                    mov.u32 %r20, %ctaid.x;
                    shl.b32 %r20, %r20, {{(int)Math.Log2(columns)}};
                    add.u32 %r20, %r20, %r5;
                    setp.ge.u32 %p16, %r20, %r2;
                    @%p16 bra DONE;
                    ld.param.u64 %rd18, [p_y0];
                    ld.param.u64 %rd19, [p_y1];
                    ld.param.u64 %rd20, [p_hidden];
                    cvta.to.global.u64 %rd18, %rd18;
                    cvta.to.global.u64 %rd19, %rd19;
                    cvta.to.global.u64 %rd20, %rd20;
                    ld.param.u32 %r21, [p_act];
                    setp.eq.u32 %p17, %r21, 1;
                    setp.eq.u32 %p18, %r21, 2;
                    mov.u32 %r32, 0;
                PR_ROW:
                    setp.ge.u32 %p16, %r32, %r1;
                    @%p16 bra DONE;
                    mad.lo.u32 %r22, %r32, %r2, %r20;
                    mul.wide.u32 %rd21, %r22, 4;
                    add.u64 %rd22, %rd21, %rd18;
                    ld.global.cg.f32 %f{{g}}, [%rd22];
                    add.u64 %rd22, %rd21, %rd19;
                    ld.global.cg.f32 %f{{u}}, [%rd22];
                    @%p17 bra PR_GELU;
                    @%p18 bra PR_RELU;
                    mul.f32 %f{{t1}}, %f{{g}}, {{F(-1.4426950408889634f)}};
                    ex2.approx.ftz.f32 %f{{t1}}, %f{{t1}};
                    add.f32 %f{{t1}}, %f{{t1}}, {{One}};
                    rcp.rn.f32 %f{{t1}}, %f{{t1}};
                    mul.f32 %f{{a}}, %f{{g}}, %f{{t1}};
                    bra PR_ACT;
                PR_GELU:
                    mul.f32 %f{{t1}}, %f{{g}}, %f{{g}};
                    mul.f32 %f{{t2}}, %f{{t1}}, %f{{g}};
                    fma.rn.f32 %f{{t2}}, %f{{t2}}, {{F(0.044715f)}}, %f{{g}};
                    mul.f32 %f{{t2}}, %f{{t2}}, {{F(0.7978845608f)}};
                    mul.f32 %f{{t3}}, %f{{t2}}, {{F(2.8853900817779268f)}};
                    ex2.approx.ftz.f32 %f{{t3}}, %f{{t3}};
                    add.f32 %f{{t3}}, %f{{t3}}, {{One}};
                    rcp.rn.f32 %f{{t3}}, %f{{t3}};
                    fma.rn.f32 %f{{t3}}, %f{{t3}}, {{F(-2f)}}, {{One}};
                    add.f32 %f{{t3}}, %f{{t3}}, {{One}};
                    mul.f32 %f{{a}}, %f{{t3}}, %f{{g}};
                    mul.f32 %f{{a}}, %f{{a}}, {{F(0.5f)}};
                    bra PR_ACT;
                PR_RELU:
                    max.f32 %f{{a}}, %f{{g}}, 0f00000000;
                PR_ACT:
                    mul.f32 %f{{a}}, %f{{a}}, %f{{u}};
                    add.u64 %rd22, %rd21, %rd20;
                    st.global.f32 [%rd22], %f{{a}};
                    add.u32 %r32, %r32, 1;
                    bra PR_ROW;
                """);
        }

        s.AppendLine("""
            DONE:
                ret;
            }
            """);
        sb.AppendLine(s.ToString());
    }

    // The normaddnorm kernels' last block (Int8Gemv, normAdd), once y is final: per row, y's RMS normalization · (ygain +
    // yoffset), sum = res + that, norm = sum's normalization · (gain + offset). Each pass is the row kernels' own: the first
    // RowThreads threads take columns tid, tid + RowThreads, … in turn (four loads in flight, their sums in that order),
    // the warp's butterfly, then warps 0 to RowThreads / 32 - 1 added in order from 0 (BlockReduce); the others only meet
    // the barriers. Multiplications and additions round on their own (.rn), as the separate kernels' stores did.
    private static string NormAddNorm(int red)
    {
        int e1 = red + 6, o1 = red + 7, nf = red + 8, acc = red + 9, sh = red + 10, ye = red + 11, yo = red + 12, inv = red + 13;
        int v = red + 14, w = red + 18, g = red + 22;                  // four values each: y (then the sum), residual, gain
        const int Unroll = 4, Stride = RowThreads * 4;                 // bytes between a thread's consecutive columns
        var s = new StringBuilder();
        s.AppendLine(CultureInfo.InvariantCulture, $$"""
                ld.param.u64 %rd18, [p_res];
                ld.param.u64 %rd19, [p_sum];
                ld.param.u64 %rd20, [p_gain];
                ld.param.u64 %rd21, [p_norm];
                ld.param.u64 %rd24, [p_ygain];
                cvta.to.global.u64 %rd18, %rd18;
                cvta.to.global.u64 %rd19, %rd19;
                cvta.to.global.u64 %rd20, %rd20;
                cvta.to.global.u64 %rd21, %rd21;
                cvta.to.global.u64 %rd24, %rd24;
                ld.param.f32 %f{{e1}}, [p_eps];
                ld.param.f32 %f{{o1}}, [p_offset];
                ld.param.f32 %f{{ye}}, [p_yeps];
                ld.param.f32 %f{{yo}}, [p_yoffset];
                cvt.rn.f32.u32 %f{{nf}}, %r2;
                shl.b32 %r22, %r7, 2;
                add.u32 %r22, %r22, %r14;
                setp.eq.u32 %p18, %r6, 0;
                setp.lt.u32 %p19, %r5, {{RowThreads}};
                mov.u32 %r32, 0;
            NA_ROW:
                setp.ge.u32 %p16, %r32, %r1;
                @%p16 bra DONE;
                mul.lo.u32 %r33, %r32, %r2;
                mul.wide.u32 %rd22, %r33, 4;
            """);

        // A pass's loop head: %r20 = this thread's next column (the row's end for threads past RowThreads), %p1-%p3 whether
        // the 2nd to 4th columns of the step exist, %rd23 the column's byte offset, %rd30 the element's.
        string Head(string label) => string.Create(CultureInfo.InvariantCulture, $"""
                selp.b32 %r20, %r5, %r2, %p19;
            {label}:
                setp.ge.u32 %p17, %r20, %r2;
                @%p17 bra {label}_END;
                add.u32 %r21, %r20, {RowThreads};
                setp.lt.u32 %p1, %r21, %r2;
                add.u32 %r21, %r20, {2 * RowThreads};
                setp.lt.u32 %p2, %r21, %r2;
                add.u32 %r21, %r20, {3 * RowThreads};
                setp.lt.u32 %p3, %r21, %r2;
                mul.wide.u32 %rd23, %r20, 4;
                add.u64 %rd30, %rd23, %rd22;
            """);

        string Tail(string label) => string.Create(CultureInfo.InvariantCulture, $"""
                add.u32 %r20, %r20, {Unroll * RowThreads};
                bra {label};
            {label}_END:
            """);

        // @%p{j} for the j-th column of a step (the first always exists inside the loop).
        static string On(int j) => j == 0 ? "" : string.Create(CultureInfo.InvariantCulture, $"@%p{j} ");

        // acc over the block as BlockReduce adds it, then inv = 1 / sqrt(acc / cols + eps) as the row kernels compute it.
        string Reduce(int eps)
        {
            var r = new StringBuilder();
            foreach (int offset in new[] { 16, 8, 4, 2, 1 })
            {
                r.AppendLine(CultureInfo.InvariantCulture, $"    shfl.sync.bfly.b32 %f{sh}, %f{acc}, {offset}, 31, 0xffffffff;");
                r.AppendLine(CultureInfo.InvariantCulture, $"    add.f32 %f{acc}, %f{acc}, %f{sh};");
            }

            r.AppendLine(CultureInfo.InvariantCulture, $"    @%p18 st.shared.f32 [%r22], %f{acc};");
            r.AppendLine("    bar.sync 0;");
            r.AppendLine(CultureInfo.InvariantCulture, $"    mov.f32 %f{acc}, {Zero};");
            for (int warp = 0; warp < RowThreads / 32; warp++)
            {
                r.AppendLine(CultureInfo.InvariantCulture, $"    ld.shared.f32 %f{sh}, [%r14+{4 * warp}];");
                r.AppendLine(CultureInfo.InvariantCulture, $"    add.f32 %f{acc}, %f{acc}, %f{sh};");
            }

            r.AppendLine(CultureInfo.InvariantCulture, $"""
                    bar.sync 0;
                    div.rn.f32 %f{inv}, %f{acc}, %f{nf};
                    add.f32 %f{inv}, %f{inv}, %f{eps};
                    sqrt.rn.f32 %f{inv}, %f{inv};
                    rcp.rn.f32 %f{inv}, %f{inv};
                """);
            return r.ToString();
        }

        // Pass 1: Σ y² (y written by the other blocks: read through L2).
        s.AppendLine(CultureInfo.InvariantCulture, $"    mov.f32 %f{acc}, {Zero};");
        s.AppendLine(Head("NA_Y"));
        s.AppendLine("    add.u64 %rd25, %rd30, %rd4;");
        for (int j = 0; j < Unroll; j++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    {On(j)}ld.global.cg.f32 %f{v + j}, [%rd25+{j * Stride}];");
        }

        for (int j = 0; j < Unroll; j++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    {On(j)}fma.rn.f32 %f{acc}, %f{v + j}, %f{v + j}, %f{acc};");
        }

        s.AppendLine(Tail("NA_Y"));
        s.AppendLine(Reduce(ye));

        // Pass 2: y = y · inv · (ygain + yoffset) and sum = res + y, both written (as rms_norm_affine_f32 in place, then
        // add_rms_norm_affine_f32), and Σ sum².
        s.AppendLine(CultureInfo.InvariantCulture, $"    mov.f32 %f{acc}, {Zero};");
        s.AppendLine(Head("NA_S"));
        s.AppendLine("""
                add.u64 %rd25, %rd30, %rd4;
                add.u64 %rd26, %rd30, %rd18;
                add.u64 %rd27, %rd23, %rd24;
                add.u64 %rd28, %rd30, %rd19;
            """);
        for (int j = 0; j < Unroll; j++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    {On(j)}ld.global.cg.f32 %f{v + j}, [%rd25+{j * Stride}];");
            s.AppendLine(CultureInfo.InvariantCulture, $"    {On(j)}ld.global.f32 %f{w + j}, [%rd26+{j * Stride}];");
            s.AppendLine(CultureInfo.InvariantCulture, $"    {On(j)}ld.global.f32 %f{g + j}, [%rd27+{j * Stride}];");
        }

        for (int j = 0; j < Unroll; j++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"""
                    {On(j)}add.f32 %f{g + j}, %f{g + j}, %f{yo};
                    {On(j)}mul.rn.f32 %f{v + j}, %f{v + j}, %f{inv};
                    {On(j)}mul.rn.f32 %f{v + j}, %f{v + j}, %f{g + j};
                    {On(j)}st.global.f32 [%rd25+{j * Stride}], %f{v + j};
                    {On(j)}add.rn.f32 %f{v + j}, %f{w + j}, %f{v + j};
                    {On(j)}st.global.f32 [%rd28+{j * Stride}], %f{v + j};
                """);
        }

        for (int j = 0; j < Unroll; j++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    {On(j)}fma.rn.f32 %f{acc}, %f{v + j}, %f{v + j}, %f{acc};");
        }

        s.AppendLine(Tail("NA_S"));
        s.AppendLine(Reduce(e1));

        // Pass 3: norm = sum · inv · (gain + offset) (the sums this thread wrote).
        s.AppendLine(Head("NA_N"));
        s.AppendLine("""
                add.u64 %rd28, %rd30, %rd19;
                add.u64 %rd27, %rd23, %rd20;
                add.u64 %rd25, %rd30, %rd21;
            """);
        for (int j = 0; j < Unroll; j++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    {On(j)}ld.global.f32 %f{v + j}, [%rd28+{j * Stride}];");
            s.AppendLine(CultureInfo.InvariantCulture, $"    {On(j)}ld.global.f32 %f{g + j}, [%rd27+{j * Stride}];");
        }

        for (int j = 0; j < Unroll; j++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"""
                    {On(j)}add.f32 %f{g + j}, %f{g + j}, %f{o1};
                    {On(j)}mul.rn.f32 %f{v + j}, %f{v + j}, %f{inv};
                    {On(j)}mul.rn.f32 %f{v + j}, %f{v + j}, %f{g + j};
                    {On(j)}st.global.f32 [%rd25+{j * Stride}], %f{v + j};
                """);
        }

        s.AppendLine(Tail("NA_N"));
        s.AppendLine("""
                add.u32 %r32, %r32, 1;
                bra NA_ROW;
            """);
        return s.ToString();
    }
}
