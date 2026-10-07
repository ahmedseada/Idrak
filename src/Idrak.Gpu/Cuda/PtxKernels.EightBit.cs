// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;

namespace Idrak.Gpu.Cuda;

/// <summary>
/// 8-bit tensor-core products: FP8 (e4m3, compute capability 8.9 and newer) or INT8 (8.0 and newer) operands, twice the
/// bfloat16 rate. Operands are quantized first into K-major byte matrices with one scale per row of a and per column of
/// b (the product's scale is then exactly scaleₐ[i] · scale_b[j], applied in the epilogue); the main loop streams them
/// with cp.async through a three-stage shared-memory pipeline.
/// </summary>
internal static partial class PtxKernels
{
    private const int EightK = 64, EightStride = EightK + 16, EightStageBytes = 2 * TensorTile * EightStride, EightStages = 3;

    /// <summary>Dynamic shared memory of the 8-bit products (bytes).</summary>
    public const int EightBitShared = EightStages * EightStageBytes;

    /// <summary>The k dimension the quantized operands are padded to (a multiple of the 8-bit products' k tile).</summary>
    public static int EightBitPaddedK(int k) => (k + EightK - 1) / EightK * EightK;

    /// <summary>The 8-bit product for the format and epilogue.</summary>
    public static string EightBitKernel(bool fp8, GemmEpilogue epilogue) =>
        $"gemm8_{(fp8 ? "e4m3" : "s8")}{epilogue switch { GemmEpilogue.Gelu => "_gelu", GemmEpilogue.GeluGradient => "_gelugrad", _ => "" }}_f32";

    private static void BuildEightBit(StringBuilder sb, bool fp8)
    {
        for (int mode = 0; mode < 3; mode++)
        {
            EightBitGemm(sb, fp8, mode);
        }

        QuantizeRows(sb, fp8);
        ColumnMaxima(sb, fp8);
        QuantizeColumns(sb, fp8);
    }

    // Largest magnitude of the format (e4m3: 448; int8: 127, symmetric).
    private static string QuantMax(bool fp8) => fp8 ? F(448f) : F(127f);

    // Four floats %x0-%x3 (already scaled) → one word of four 8-bit values (x0 in the low byte) in %r40.
    private static string PackFour(bool fp8) => fp8
        ? """
            cvt.rn.satfinite.e4m3x2.f32 %hs0, %x1, %x0;
            cvt.rn.satfinite.e4m3x2.f32 %hs1, %x3, %x2;
            mov.b32 %r40, {%hs0, %hs1};
            """
        : """
            cvt.rni.s32.f32 %r41, %x0;
            cvt.rni.s32.f32 %r42, %x1;
            cvt.rni.s32.f32 %r43, %x2;
            cvt.rni.s32.f32 %r44, %x3;
            min.s32 %r41, %r41, 127;
            max.s32 %r41, %r41, -127;
            min.s32 %r42, %r42, 127;
            max.s32 %r42, %r42, -127;
            min.s32 %r43, %r43, 127;
            max.s32 %r43, %r43, -127;
            min.s32 %r44, %r44, 127;
            max.s32 %r44, %r44, -127;
            and.b32 %r41, %r41, 255;
            and.b32 %r42, %r42, 255;
            and.b32 %r43, %r43, 255;
            and.b32 %r44, %r44, 255;
            shl.b32 %r42, %r42, 8;
            shl.b32 %r43, %r43, 16;
            shl.b32 %r44, %r44, 24;
            or.b32 %r40, %r41, %r42;
            or.b32 %r40, %r40, %r43;
            or.b32 %r40, %r40, %r44;
            """;

    // out[r, 0 … ldo) = quantized x[r, :] (zero past `cols`), scale[r] = amax / max; one block of 256 threads per row.
    private static void QuantizeRows(StringBuilder sb, bool fp8)
    {
        string name = $"quant_rows_{(fp8 ? "e4m3" : "s8")}";
        sb.AppendLine($$"""
            .visible .entry {{name}}(
                .param .u64 p_x, .param .u64 p_out, .param .u64 p_scale, .param .u32 p_ld, .param .u32 p_ldo, .param .u32 p_rows, .param .u32 p_cols
            )
            {
                .reg .pred %p<8>;
                .reg .b32 %r<48>;
                .reg .b64 %rd<16>;
                .reg .f32 %f<16>;
                .reg .f32 %x<4>;
                .reg .b16 %hs<2>;
                .shared .align 4 .f32 {{name}}_max[8];
                mov.u32 %r1, %ctaid.x;
                ld.param.u32 %r2, [p_rows];
                setp.ge.u32 %p1, %r1, %r2;
                @%p1 bra DONE;
                ld.param.u64 %rd1, [p_x];
                ld.param.u64 %rd2, [p_out];
                ld.param.u64 %rd3, [p_scale];
                cvta.to.global.u64 %rd1, %rd1;
                cvta.to.global.u64 %rd2, %rd2;
                cvta.to.global.u64 %rd3, %rd3;
                ld.param.u32 %r3, [p_ld];
                ld.param.u32 %r4, [p_ldo];
                ld.param.u32 %r5, [p_cols];
                mov.u32 %r6, %tid.x;
                mul.wide.u32 %rd4, %r1, %r3;
                shl.b64 %rd4, %rd4, 2;
                add.u64 %rd4, %rd4, %rd1;
                mov.f32 %f1, 0f00000000;
                mov.u32 %r7, %r6;
            AMAX:
                setp.ge.u32 %p2, %r7, %r5;
                @%p2 bra AMAX_END;
                mul.wide.u32 %rd5, %r7, 4;
                add.u64 %rd5, %rd5, %rd4;
                ld.global.f32 %f2, [%rd5];
                abs.f32 %f2, %f2;
                max.f32 %f1, %f1, %f2;
                add.u32 %r7, %r7, 256;
                bra AMAX;
            AMAX_END:
                shfl.sync.bfly.b32 %f2, %f1, 16, 31, 0xffffffff;
                max.f32 %f1, %f1, %f2;
                shfl.sync.bfly.b32 %f2, %f1, 8, 31, 0xffffffff;
                max.f32 %f1, %f1, %f2;
                shfl.sync.bfly.b32 %f2, %f1, 4, 31, 0xffffffff;
                max.f32 %f1, %f1, %f2;
                shfl.sync.bfly.b32 %f2, %f1, 2, 31, 0xffffffff;
                max.f32 %f1, %f1, %f2;
                shfl.sync.bfly.b32 %f2, %f1, 1, 31, 0xffffffff;
                max.f32 %f1, %f1, %f2;
                and.b32 %r8, %r6, 31;
                shr.u32 %r9, %r6, 5;
                mov.u32 %r10, {{name}}_max;
                setp.eq.u32 %p3, %r8, 0;
                shl.b32 %r11, %r9, 2;
                add.u32 %r11, %r11, %r10;
                @%p3 st.shared.f32 [%r11], %f1;
                bar.sync 0;
                ld.shared.f32 %f1, [%r10];
                ld.shared.f32 %f2, [%r10+4];
                max.f32 %f1, %f1, %f2;
                ld.shared.f32 %f2, [%r10+8];
                max.f32 %f1, %f1, %f2;
                ld.shared.f32 %f2, [%r10+12];
                max.f32 %f1, %f1, %f2;
                ld.shared.f32 %f2, [%r10+16];
                max.f32 %f1, %f1, %f2;
                ld.shared.f32 %f2, [%r10+20];
                max.f32 %f1, %f1, %f2;
                ld.shared.f32 %f2, [%r10+24];
                max.f32 %f1, %f1, %f2;
                ld.shared.f32 %f2, [%r10+28];
                max.f32 %f1, %f1, %f2;
                setp.gt.f32 %p4, %f1, 0f00000000;
                div.rn.f32 %f3, %f1, {{QuantMax(fp8)}};
                selp.f32 %f3, %f3, 0f3F800000, %p4;
                rcp.rn.f32 %f4, %f3;
                setp.eq.u32 %p5, %r6, 0;
                mul.wide.u32 %rd6, %r1, 4;
                add.u64 %rd6, %rd6, %rd3;
                @%p5 st.global.f32 [%rd6], %f3;
                mul.wide.u32 %rd7, %r1, %r4;
                add.u64 %rd7, %rd7, %rd2;
                shl.b32 %r12, %r6, 2;
            QUANT:
                setp.ge.u32 %p2, %r12, %r4;
                @%p2 bra DONE;
            {{string.Concat(Enumerable.Range(0, 4).Select(j => $"""
                add.u32 %r13, %r12, {j};
                setp.lt.u32 %p6, %r13, %r5;
                mov.f32 %x{j}, 0f00000000;
                mul.wide.u32 %rd8, %r13, 4;
                add.u64 %rd8, %rd8, %rd4;
                @%p6 ld.global.f32 %x{j}, [%rd8];
                mul.f32 %x{j}, %x{j}, %f4;

            """))}}
            {{PackFour(fp8)}}
                cvt.u64.u32 %rd9, %r12;
                add.u64 %rd9, %rd9, %rd7;
                st.global.b32 [%rd9], %r40;
                add.u32 %r12, %r12, 1024;
                bra QUANT;
            DONE:
                ret;
            }
            """);
    }

    // amax[j] = max(amax[j], max over a chunk of rows of |x[r, j]|) (float bits, atomically; zeroed first).
    // Grid (⌈cols / 256⌉, ⌈rows / chunk⌉), 256 threads.
    private static void ColumnMaxima(StringBuilder sb, bool fp8) => sb.AppendLine($$"""
        .visible .entry absmax_cols_{{(fp8 ? "e4m3" : "s8")}}(
            .param .u64 p_x, .param .u64 p_amax, .param .u32 p_ld, .param .u32 p_rows, .param .u32 p_cols, .param .u32 p_chunk
        )
        {
            .reg .pred %p<4>;
            .reg .b32 %r<16>;
            .reg .b64 %rd<8>;
            .reg .f32 %f<4>;
            ld.param.u64 %rd1, [p_x];
            ld.param.u64 %rd2, [p_amax];
            cvta.to.global.u64 %rd1, %rd1;
            cvta.to.global.u64 %rd2, %rd2;
            ld.param.u32 %r1, [p_ld];
            ld.param.u32 %r2, [p_rows];
            ld.param.u32 %r3, [p_cols];
            ld.param.u32 %r4, [p_chunk];
            mov.u32 %r5, %ctaid.x;
            mov.u32 %r6, %tid.x;
            mad.lo.u32 %r7, %r5, 256, %r6;
            setp.ge.u32 %p1, %r7, %r3;
            @%p1 bra DONE;
            mov.u32 %r8, %ctaid.y;
            mul.lo.u32 %r9, %r8, %r4;
            add.u32 %r10, %r9, %r4;
            min.u32 %r10, %r10, %r2;
            mov.f32 %f1, 0f00000000;
        ROWS:
            setp.ge.u32 %p2, %r9, %r10;
            @%p2 bra ROWS_END;
            mul.wide.u32 %rd3, %r9, %r1;
            cvt.u64.u32 %rd4, %r7;
            add.u64 %rd3, %rd3, %rd4;
            shl.b64 %rd3, %rd3, 2;
            add.u64 %rd3, %rd3, %rd1;
            ld.global.f32 %f2, [%rd3];
            abs.f32 %f2, %f2;
            max.f32 %f1, %f1, %f2;
            add.u32 %r9, %r9, 1;
            bra ROWS;
        ROWS_END:
            mov.b32 %r11, %f1;
            mul.wide.u32 %rd5, %r7, 4;
            add.u64 %rd5, %rd5, %rd2;
            atom.global.max.u32 %r12, [%rd5], %r11;
        DONE:
            ret;
        }
        """);

    // out[j, k] = quantized x[k, j] (k-major bytes, rows of ldo, zero past `rows`), scale[j] = amax[j] / max: 32 × 32 tiles
    // through shared memory. Grid (⌈cols / 32⌉, ldo / 32), block 32 × 8. When p_record is not 0, the column maxima of x
    // are also recorded there (atomic max of float bits, zeroed first): with maxima kept from an earlier pass in p_amax
    // (delayed scaling), one read of x quantizes it, with twice the kept maxima as headroom, and measures it for the next
    // time; values beyond twice the kept maxima saturate. p_only (not 0) turns the kernel into that pass's correction: a
    // block whose 32 columns all have amax[j] ≤ 2·only[j] (only = the kept maxima: nothing saturated) returns before
    // reading x, the others quantize again with the recorded maxima (no headroom). Blocks take tiles ctaid.y, +
    // gridDim.y, … so the correction can run on a few blocks per strip.
    private static void QuantizeColumns(StringBuilder sb, bool fp8)
    {
        string name = $"quant_cols_{(fp8 ? "e4m3" : "s8")}";
        var s = new StringBuilder();
        s.AppendLine($$"""
            .visible .entry {{name}}(
                .param .u64 p_x, .param .u64 p_out, .param .u64 p_amax, .param .u64 p_scale, .param .u32 p_ld, .param .u32 p_ldo,
                .param .u32 p_rows, .param .u32 p_cols, .param .u64 p_record, .param .u64 p_only
            )
            {
                .reg .pred %p<12>;
                .reg .b32 %r<48>;
                .reg .b64 %rd<16>;
                .reg .f32 %f<16>;
                .reg .f32 %x<4>;
                .reg .b16 %hs<2>;
                .shared .align 4 .f32 {{name}}_t[1056];
                ld.param.u64 %rd1, [p_x];
                ld.param.u64 %rd2, [p_out];
                ld.param.u64 %rd3, [p_amax];
                ld.param.u64 %rd4, [p_scale];
                ld.param.u64 %rd12, [p_record];
                setp.ne.u64 %p8, %rd12, 0;
                cvta.to.global.u64 %rd1, %rd1;
                cvta.to.global.u64 %rd2, %rd2;
                cvta.to.global.u64 %rd3, %rd3;
                cvta.to.global.u64 %rd4, %rd4;
                cvta.to.global.u64 %rd12, %rd12;
                ld.param.u32 %r1, [p_ld];
                ld.param.u32 %r2, [p_ldo];
                ld.param.u32 %r3, [p_rows];
                ld.param.u32 %r4, [p_cols];
                mov.u32 %r5, %tid.x;
                mov.u32 %r6, %tid.y;
                mov.u32 %r9, {{name}}_t;
            """);
        s.AppendLine("""
                mov.u32 %r7, %ctaid.x;
                shl.b32 %r7, %r7, 5;
                mov.u32 %r8, %ctaid.y;
                shl.b32 %r8, %r8, 5;
                ld.param.u64 %rd15, [p_only];
                setp.ne.u64 %p9, %rd15, 0;
                @!%p9 bra WHOLE;
                cvta.to.global.u64 %rd15, %rd15;
                add.u32 %r23, %r7, %r5;
                setp.lt.u32 %p10, %r23, %r4;
                mul.wide.u32 %rd13, %r23, 4;
                add.u64 %rd14, %rd13, %rd3;
                add.u64 %rd13, %rd13, %rd15;
                mov.f32 %f10, 0f00000000;
                mov.f32 %f11, 0f00000000;
                @%p10 ld.global.f32 %f10, [%rd14];
                @%p10 ld.global.f32 %f11, [%rd13];
                add.f32 %f11, %f11, %f11;
                setp.gt.f32 %p10, %f10, %f11;
                bar.red.or.pred %p11, 0, %p10;
                @!%p11 bra DONE;
            WHOLE:
            YTILE:
            """);

        for (int i = 0; i < 4; i++)
        {
            s.AppendLine($"""
                    add.u32 %r10, %r6, {8 * i};
                    add.u32 %r11, %r8, %r10;
                    add.u32 %r12, %r7, %r5;
                    setp.lt.u32 %p1, %r11, %r3;
                    setp.lt.and.u32 %p1, %r12, %r4, %p1;
                    mul.wide.u32 %rd5, %r11, %r1;
                    cvt.u64.u32 %rd6, %r12;
                    add.u64 %rd5, %rd5, %rd6;
                    shl.b64 %rd5, %rd5, 2;
                    add.u64 %rd5, %rd5, %rd1;
                    mov.f32 %f1, 0f00000000;
                    @%p1 ld.global.f32 %f1, [%rd5];
                    mul.lo.u32 %r13, %r10, 33;
                    add.u32 %r13, %r13, %r5;
                    shl.b32 %r13, %r13, 2;
                    add.u32 %r13, %r13, %r9;
                    st.shared.f32 [%r13], %f1;
                """);
        }

        s.AppendLine($$"""
                bar.sync 0;
                add.u32 %r14, %r7, %r5;
                setp.ge.u32 %p2, %r14, %r4;
                @%p2 bra YTILE_DONE;
                mul.wide.u32 %rd7, %r14, 4;
                add.u64 %rd8, %rd7, %rd3;
                ld.global.f32 %f2, [%rd8];
                @%p8 add.f32 %f2, %f2, %f2;
                setp.gt.f32 %p3, %f2, 0f00000000;
                div.rn.f32 %f3, %f2, {{QuantMax(fp8)}};
                selp.f32 %f3, %f3, 0f3F800000, %p3;
                rcp.rn.f32 %f4, %f3;
                or.b32 %r15, %r6, %r8;
                setp.eq.u32 %p4, %r15, 0;
                add.u64 %rd9, %rd7, %rd4;
                @%p4 st.global.f32 [%rd9], %f3;
                shl.b32 %r16, %r6, 2;
            """);
        // Recording: thread (tx, 0) takes the maximum of its column over the tile's 32 rows.
        s.AppendLine("""
                setp.eq.u32 %p5, %r6, 0;
                and.pred %p5, %p5, %p8;
                @!%p5 bra RECORDED;
                shl.b32 %r20, %r5, 2;
                add.u32 %r20, %r20, %r9;
                mov.f32 %f5, 0f00000000;
                mov.u32 %r21, 0;
            REC:
                ld.shared.f32 %f6, [%r20];
                abs.f32 %f6, %f6;
                max.f32 %f5, %f5, %f6;
                add.u32 %r20, %r20, 132;
                add.u32 %r21, %r21, 1;
                setp.lt.u32 %p6, %r21, 32;
                @%p6 bra REC;
                mov.b32 %r22, %f5;
                add.u64 %rd13, %rd7, %rd12;
                red.global.max.u32 [%rd13], %r22;
            RECORDED:
            """);

        for (int j = 0; j < 4; j++)
        {
            s.AppendLine($"""
                    add.u32 %r17, %r16, {j};
                    mul.lo.u32 %r17, %r17, 33;
                    add.u32 %r17, %r17, %r5;
                    shl.b32 %r17, %r17, 2;
                    add.u32 %r17, %r17, %r9;
                    ld.shared.f32 %x{j}, [%r17];
                    mul.f32 %x{j}, %x{j}, %f4;
                """);
        }

        s.AppendLine(PackFour(fp8));
        s.AppendLine("""
                mul.wide.u32 %rd10, %r14, %r2;
                add.u32 %r18, %r8, %r16;
                cvt.u64.u32 %rd11, %r18;
                add.u64 %rd10, %rd10, %rd11;
                add.u64 %rd10, %rd10, %rd2;
                st.global.b32 [%rd10], %r40;
            """);
        // Tiles ctaid.y, + gridDim.y, … (one per block unless the correction pass runs on a few blocks per strip).
        s.AppendLine("""
            YTILE_DONE:
                bar.sync 0;
                mov.u32 %r24, %nctaid.y;
                shl.b32 %r24, %r24, 5;
                add.u32 %r8, %r8, %r24;
                setp.lt.u32 %p5, %r8, %r2;
                @%p5 bra YTILE;
            """);

        s.AppendLine("""
            DONE:
                ret;
            }
            """);
        sb.Append(s);
        sb.AppendLine();
    }

    // c = beta·c + f(scaleₐ[i] · scale_b[j] · Σ_k a8[i, k] · b8[j, k]) for a8 [m][lda], b8 [n][ldb] (bytes, k-major, k a
    // multiple of 64 with zero padding), epilogue as the bfloat16 products (bias, modes, strided c, aux).
    // 256 threads: warps 2 × 4 over the 128 × 128 tile (64 × 32 each); three cp.async stages of 64 k.
    private static void EightBitGemm(StringBuilder sb, bool fp8, int mode)
    {
        string name = EightBitKernel(fp8, (GemmEpilogue)mode);
        string mma = fp8 ? "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32" : "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32";
        string acc = fp8 ? "%c" : "%ci";
        var s = new StringBuilder();
        s.AppendLine($$"""
            .extern .shared .align 16 .b8 {{name}}_smem[];
            .visible .entry {{name}}(
                .param .u64 p_a, .param .u64 p_b, .param .u64 p_c,
                .param .u32 p_m, .param .u32 p_n, .param .u32 p_k, .param .f32 p_beta,
                .param .u64 p_sa, .param .u64 p_sb, .param .u64 p_bias,
                .param .u32 p_lda, .param .u32 p_ldb, .param .u32 p_ldc, .param .u64 p_aux
            )
            {
                .reg .pred %p<16>;
                .reg .pred %pbeta, %peven, %qnot;
                .reg .pred %q<12>;
                .reg .pred %pbias, %paux;
                .reg .pred %pa<2>;
                .reg .pred %pb<2>;
                .reg .f32 %e<8>;
                .reg .f32 %h<8>;
                .reg .f32 %t<8>;
                .reg .f32 %bias<8>;
                .reg .f32 %c<64>;
                .reg .b32 %ci<64>;
                .reg .b32 %fa<16>;
                .reg .b32 %fb<8>;
                .reg .f32 %f<4>;
                .reg .f32 %beta;
                .reg .b32 %r<80>;
                .reg .b64 %rd<48>;
                ld.param.u64 %rd1, [p_a];
                ld.param.u64 %rd2, [p_b];
                ld.param.u64 %rd3, [p_c];
                cvta.to.global.u64 %rd1, %rd1;
                cvta.to.global.u64 %rd2, %rd2;
                cvta.to.global.u64 %rd3, %rd3;
                ld.param.u32 %r1, [p_m];
                ld.param.u32 %r2, [p_n];
                ld.param.u32 %r3, [p_k];
                ld.param.f32 %beta, [p_beta];
                setp.ne.f32 %pbeta, %beta, 0f00000000;
                ld.param.u32 %r58, [p_lda];
                ld.param.u32 %r59, [p_ldb];
                ld.param.u32 %r60, [p_ldc];
                mov.u32 %r4, %tid.x;
                and.b32 %r5, %r4, 31;
                shr.u32 %r6, %r4, 5;
                and.b32 %r7, %r6, 1;
                shr.u32 %r8, %r6, 1;
                mov.u32 %r50, %nctaid.x;
                mov.u32 %r51, %nctaid.y;
                mov.u32 %r52, %ctaid.y;
                mov.u32 %r53, %ctaid.x;
                mad.lo.u32 %r52, %r52, %r50, %r53;
                shl.b32 %r54, %r50, 3;
                div.u32 %r55, %r52, %r54;
                shl.b32 %r55, %r55, 3;
                sub.u32 %r56, %r51, %r55;
                min.u32 %r56, %r56, 8;
                rem.u32 %r57, %r52, %r54;
                rem.u32 %r9, %r57, %r56;
                add.u32 %r9, %r9, %r55;
                div.u32 %r10, %r57, %r56;
                shl.b32 %r9, %r9, 7;
                shl.b32 %r10, %r10, 7;
                mov.u32 %r11, {{name}}_smem;
            """);
        for (int i = 0; i < 64; i++)
        {
            s.AppendLine(fp8 ? $"    mov.f32 %c{i}, 0f00000000;" : $"    mov.b32 %ci{i}, 0;");
        }

        // Copy coordinates: chunk (16 bytes) %r12 of rows (tid >> 2) and (tid >> 2) + 64 of each operand's tile.
        s.AppendLine("""
                and.b32 %r12, %r4, 3;
                shl.b32 %r12, %r12, 4;
                shr.u32 %r13, %r4, 2;
            """);
        for (int i = 0; i < 2; i++)
        {
            s.AppendLine($"""
                    add.u32 %r20, %r13, {64 * i};
                    add.u32 %r21, %r9, %r20;
                    setp.lt.u32 %pa{i}, %r21, %r1;
                    mul.wide.u32 %rd{10 + i}, %r21, %r58;
                    cvt.u64.u32 %rd14, %r12;
                    add.u64 %rd{10 + i}, %rd{10 + i}, %rd14;
                    add.u64 %rd{10 + i}, %rd{10 + i}, %rd1;
                    add.u32 %r22, %r10, %r20;
                    setp.lt.u32 %pb{i}, %r22, %r2;
                    mul.wide.u32 %rd{12 + i}, %r22, %r59;
                    add.u64 %rd{12 + i}, %rd{12 + i}, %rd14;
                    add.u64 %rd{12 + i}, %rd{12 + i}, %rd2;
                    mul.lo.u32 %r{24 + i}, %r20, {EightStride};
                    add.u32 %r{24 + i}, %r{24 + i}, %r12;
                """);
        }

        // Issues the copies of k tile %r17 into stage offset `stage`: 16 bytes each, 0 (zero fill) past m, n or k.
        void Issue(string stage)
        {
            s.AppendLine($"""
                    add.u32 %r26, %r17, %r12;
                    setp.lt.u32 %p1, %r26, %r3;
                    cvt.u64.u32 %rd15, %r17;
                    add.u32 %r27, %r11, {stage};
                """);
            for (int i = 0; i < 2; i++)
            {
                s.AppendLine($"""
                        and.pred %p2, %p1, %pa{i};
                        selp.u32 %r28, 16, 0, %p2;
                        add.u64 %rd16, %rd{10 + i}, %rd15;
                        add.u32 %r29, %r27, %r{24 + i};
                        cp.async.cg.shared.global [%r29], [%rd16], 16, %r28;
                        and.pred %p2, %p1, %pb{i};
                        selp.u32 %r28, 16, 0, %p2;
                        add.u64 %rd16, %rd{12 + i}, %rd15;
                        add.u32 %r29, %r29, {TensorTile * EightStride};
                        cp.async.cg.shared.global [%r29], [%rd16], 16, %r28;
                    """);
            }
        }

        // ldmatrix lane offsets: A rows (l & 15) at byte 16 (l >> 4); B rows (l & 7) + 8 (l >> 4) at byte 16 ((l >> 3) & 1).
        s.AppendLine($"""
                and.b32 %r30, %r5, 15;
                shl.b32 %r31, %r7, 6;
                add.u32 %r30, %r30, %r31;
                mul.lo.u32 %r30, %r30, {EightStride};
                shr.u32 %r31, %r5, 4;
                shl.b32 %r31, %r31, 4;
                add.u32 %r30, %r30, %r31;
                and.b32 %r32, %r5, 7;
                shr.u32 %r33, %r5, 4;
                shl.b32 %r33, %r33, 3;
                add.u32 %r32, %r32, %r33;
                shl.b32 %r33, %r8, 5;
                add.u32 %r32, %r32, %r33;
                mul.lo.u32 %r32, %r32, {EightStride};
                shr.u32 %r33, %r5, 3;
                and.b32 %r33, %r33, 1;
                shl.b32 %r33, %r33, 4;
                add.u32 %r32, %r32, %r33;
                add.u32 %r32, %r32, {TensorTile * EightStride};
                add.u32 %r14, %r3, {EightK - 1};
                shr.u32 %r14, %r14, 6;
                mov.u32 %r17, 0;
            """);
        // Prologue: stages 0 and 1.
        s.AppendLine("    setp.lt.u32 %p3, 0, %r14;\n    @!%p3 bra PRO0;");
        Issue("0");
        s.AppendLine($"""
                PRO0:
                    cp.async.commit_group;
                    mov.u32 %r17, {EightK};
                    setp.lt.u32 %p3, 1, %r14;
                    @!%p3 bra PRO1;
                """);
        Issue(EightStageBytes.ToString(System.Globalization.CultureInfo.InvariantCulture));
        s.AppendLine($"""
                PRO1:
                    cp.async.commit_group;
                    mov.u32 %r15, 0;
                    mov.u32 %r16, {2 * EightStageBytes};
                    mov.u32 %r17, {2 * EightK};
                    mov.u32 %r18, 0;
                LOOP:
                    setp.ge.u32 %p4, %r18, %r14;
                    @%p4 bra LOOP_END;
                    cp.async.wait_group 1;
                    bar.sync 0;
                    add.u32 %r19, %r18, 2;
                    setp.lt.u32 %p5, %r19, %r14;
                    @!%p5 bra NO_ISSUE;
                """);
        Issue("%r16");
        s.AppendLine("""
            NO_ISSUE:
                cp.async.commit_group;
                add.u32 %r34, %r11, %r15;
                add.u32 %r35, %r34, %r30;
                add.u32 %r36, %r34, %r32;
            """);
        for (int ks = 0; ks < 2; ks++)
        {
            for (int mt = 0; mt < 4; mt++)
            {
                s.AppendLine($"    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%fa{4 * mt}, %fa{4 * mt + 1}, %fa{4 * mt + 2}, %fa{4 * mt + 3}}}, [%r35+{mt * 16 * EightStride + ks * 32}];");
            }

            for (int np = 0; np < 2; np++)
            {
                s.AppendLine($"    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%fb{4 * np}, %fb{4 * np + 1}, %fb{4 * np + 2}, %fb{4 * np + 3}}}, [%r36+{np * 16 * EightStride + ks * 32}];");
            }

            for (int mt = 0; mt < 4; mt++)
            {
                for (int nt = 0; nt < 4; nt++)
                {
                    int c = (mt * 4 + nt) * 4, fb = (nt / 2) * 4 + (nt % 2) * 2;
                    s.AppendLine($"    {mma} {{{acc}{c}, {acc}{c + 1}, {acc}{c + 2}, {acc}{c + 3}}}, "
                                 + $"{{%fa{4 * mt}, %fa{4 * mt + 1}, %fa{4 * mt + 2}, %fa{4 * mt + 3}}}, {{%fb{fb}, %fb{fb + 1}}}, "
                                 + $"{{{acc}{c}, {acc}{c + 1}, {acc}{c + 2}, {acc}{c + 3}}};");
                }
            }
        }

        s.AppendLine($"""
                add.u32 %r15, %r15, {EightStageBytes};
                setp.eq.u32 %p6, %r15, {EightBitShared};
                selp.u32 %r15, 0, %r15, %p6;
                add.u32 %r16, %r16, {EightStageBytes};
                setp.eq.u32 %p6, %r16, {EightBitShared};
                selp.u32 %r16, 0, %r16, %p6;
                add.u32 %r17, %r17, {EightK};
                add.u32 %r18, %r18, 1;
                bra LOOP;
            LOOP_END:
                cp.async.wait_group 0;
            """);

        EmitTensorEpilogue(s, mode, e =>
        {
            // Accumulators (int8: converted) · scaleₐ[row] · scale_b[column]: rows %r42 + 16 mt + 8 half in %h, columns
            // %r43 + 8 nt + j in %e.
            if (!fp8)
            {
                for (int i = 0; i < 64; i++)
                {
                    e.AppendLine($"    cvt.rn.f32.s32 %c{i}, %ci{i};");
                }
            }

            e.AppendLine("""
                    ld.param.u64 %rd30, [p_sa];
                    ld.param.u64 %rd31, [p_sb];
                    cvta.to.global.u64 %rd30, %rd30;
                    cvta.to.global.u64 %rd31, %rd31;
                """);
            for (int r = 0; r < 8; r++)
            {
                e.AppendLine($"""
                        mov.f32 %h{r}, 0f00000000;
                        add.u32 %r46, %r42, {r * 8};
                        setp.lt.u32 %p9, %r46, %r1;
                        mul.wide.u32 %rd20, %r46, 4;
                        add.u64 %rd19, %rd30, %rd20;
                        @%p9 ld.global.f32 %h{r}, [%rd19];
                    """);
            }

            for (int col = 0; col < 8; col++)
            {
                e.AppendLine($"""
                        mov.f32 %e{col}, 0f00000000;
                        add.u32 %r46, %r43, {(col / 2) * 8 + col % 2};
                        setp.lt.u32 %p9, %r46, %r2;
                        mul.wide.u32 %rd20, %r46, 4;
                        add.u64 %rd19, %rd31, %rd20;
                        @%p9 ld.global.f32 %e{col}, [%rd19];
                    """);
            }

            for (int mt = 0; mt < 4; mt++)
            {
                for (int nt = 0; nt < 4; nt++)
                {
                    for (int v = 0; v < 4; v++)
                    {
                        int c = (mt * 4 + nt) * 4 + v, row = mt * 2 + v / 2, col = nt * 2 + v % 2;
                        e.AppendLine($"    mul.f32 %c{c}, %c{c}, %h{row};");
                        e.AppendLine($"    mul.f32 %c{c}, %c{c}, %e{col};");
                    }
                }
            }
        });
        s.AppendLine("""
                ret;
            }
            """);
        sb.Append(s);
        sb.AppendLine();
    }
}
