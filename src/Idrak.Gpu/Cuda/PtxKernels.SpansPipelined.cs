// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;

namespace Idrak.Gpu.Cuda;

// Attention over one range of keys per query row on bfloat16 tensor cores, pipelined (CudaBackend.Spans.cs): the
// tensor-core span kernel of PtxKernels.SpansTiled.cs with its keys and values read as bfloat16 from a copy made once per
// call, and copied into shared memory with cp.async two tiles deep, so the next tile's copies run while the tensor cores
// work on this one (the single-stage kernel loads, rounds and stores every tile of keys and values in every block, then
// waits for them before any product).
//
// - span_pack_bf16: keys and values (float32, through a row layout) to bfloat16 [kvHeads, keyRows, V] (V = the head size
//   rounded up to 8; zeros past the head size), rounded as the single-stage kernel rounds them as it stages them
//   (cvt.rn.bf16x2). Grid: x = ⌈keyRows · V / 4 / 256⌉, y = kvHeads, z = 2 (keys, then values); 256 threads.
//
// - attention_spans_tcp_d{V}_w{W}: W warps take 16·W query rows of one head (their Q fragments in registers, staged as
//   the single-stage kernel stages them), the keys in tiles of 64: S = Q·Kᵀ over D = V rounded up to 16 (the copies fill
//   the columns past V with zeros), O += P·V over V / 8 tiles of 8 columns (9 at head size 72, where the single-stage
//   kernel computes 10), online softmax in the log2 domain. The same bfloat16 operands, products in the same order and the
//   same softmax as the single-stage kernel: the same outputs bit for bit. Dynamic shared memory: two stages of a key tile
//   and a value tile (64 rows of D + 8 bfloat16 each); the block's queries are staged in the second stage before the loop.
//   Queries and outputs go through a row layout: head hh = b·p_hq + h, row i of q at q + b·p_qb + h·p_qh + i·p_qr
//   (floats), of the output at y + b·p_yb + h·p_yh + i·p_yr; the log-sum-exp stays [heads, rows]. Contiguous
//   [heads, rows, dim] tensors: p_hq = 1, p_qb = rows·dim, p_qh = 0, p_qr = dim.
//
// Ranges, grouped heads and tables as the other span kernels: query row i of head h sees keys st[t·rows + i] ≤ c <
// en[t·rows + i] (clamped to [0, keyRows]), t = h / headsPerTable, key/value head h / group; an empty range gives zeros and
// a log-sum-exp of -∞. Head sizes multiples of 4 up to 128 (no soft-cap). Grid: x = ⌈rows / 16·W⌉, y = heads.
internal static partial class PtxKernels
{
    /// <summary>The packing kernel of the pipelined span modules.</summary>
    public const string SpanPackName = "span_pack_bf16";

    /// <summary>Threads per block of <see cref="SpanPackName"/>.</summary>
    public const int SpanPackThreads = 256;

    /// <summary>
    /// The head size of the pipelined span kernel for <paramref name="dim"/> (a multiple of 8: the outputs' tiles; the
    /// scores' k steps round it up to 16), or 0 when it does not take it.
    /// </summary>
    public static int SpanPipelinedDim(int dim) => dim > 0 && dim % 4 == 0 && dim <= FlashMaxDim ? (dim + 7) / 8 * 8 : 0;

    /// <summary>The pipelined span kernel's name for head size <paramref name="v"/> (a multiple of 8) and <paramref name="warps"/> warps.</summary>
    public static string SpanPipelinedName(int v, int warps) => string.Create(CultureInfo.InvariantCulture, $"attention_spans_tcp_d{v}_w{warps}");

    /// <summary>
    /// Dynamic shared memory of the pipelined span kernels for head size <paramref name="v"/> (bytes): two stages of a key
    /// and a value tile of 64 rows (the queries of a block of up to 8 warps fit one stage).
    /// </summary>
    public static int SpanPipelinedShared(int v) => 4 * SpanTensorKeys * FlashStride((v + 15) / 16 * 16);

    /// <summary>
    /// The pipelined span module for head size <paramref name="v"/> (PTX 7.0, sm_80): the packing kernel, then a span
    /// kernel per warp count of <see cref="SpanTensorWarps"/>.
    /// </summary>
    public static (string Name, string[] Kernels, string Source) SpanPipelinedModule(int v) =>
        Module(string.Create(CultureInfo.InvariantCulture, $"attention spans tcp d{v}"), sb =>
        {
            sb.AppendLine(CultureInfo.InvariantCulture, $".extern .shared .align 16 .b8 spans_tcp_d{v}_smem[];");
            sb.AppendLine();
            SpanPack(sb);
            foreach (int warps in SpanTensorWarps)
            {
                SpanPipelined(sb, v, warps);
            }
        });

    // The row layout of queries and outputs (after SpanParameters).
    private const string SpanLayoutParameters =
        ".param .u32 p_hq, .param .u32 p_qb, .param .u32 p_qh, .param .u32 p_qr, .param .u32 p_yb, .param .u32 p_yh, .param .u32 p_yr";

    // Keys and values to bfloat16 rows of p_quads·4 values: key/value head g = b·p_hk + h, row c read at src + b·p_kb +
    // h·p_kh + c·p_kr (floats; four at a time, so every stride is a multiple of 4), written at out + z·(heads·keys·quads·8)
    // + ((g·keys + c)·quads + quad)·8 bytes; columns at or past p_dim are zeros.
    private static void SpanPack(StringBuilder sb) => sb.AppendLine(CultureInfo.InvariantCulture, $$"""
        .visible .entry {{SpanPackName}}(
            .param .u64 p_k, .param .u64 p_v, .param .u64 p_out, .param .u32 p_keys, .param .u32 p_dim, .param .u32 p_quads,
            .param .u32 p_hk, .param .u32 p_kb, .param .u32 p_kh, .param .u32 p_kr
        )
        .maxntid {{SpanPackThreads}}, 1, 1
        {
            .reg .pred %p<4>;
            .reg .b32 %r<24>;
            .reg .b64 %rd<12>;
            .reg .f32 %f<4>;
            ld.param.u32 %r1, [p_keys];
            ld.param.u32 %r2, [p_dim];
            ld.param.u32 %r3, [p_quads];
            mov.u32 %r4, %ctaid.x;
            mov.u32 %r5, %tid.x;
            mad.lo.u32 %r6, %r4, {{SpanPackThreads}}, %r5;
            mul.lo.u32 %r7, %r1, %r3;
            setp.ge.u32 %p1, %r6, %r7;
            @%p1 bra DONE;
            div.u32 %r8, %r6, %r3;
            mul.lo.u32 %r9, %r8, %r3;
            sub.u32 %r9, %r6, %r9;
            mov.u32 %r10, %ctaid.y;
            mov.u32 %r11, %ctaid.z;
            ld.param.u64 %rd1, [p_k];
            ld.param.u64 %rd2, [p_v];
            setp.ne.u32 %p2, %r11, 0;
            selp.b64 %rd1, %rd2, %rd1, %p2;
            cvta.to.global.u64 %rd1, %rd1;
            ld.param.u32 %r12, [p_hk];
            ld.param.u32 %r13, [p_kb];
            ld.param.u32 %r14, [p_kh];
            ld.param.u32 %r15, [p_kr];
            div.u32 %r16, %r10, %r12;
            mul.lo.u32 %r17, %r16, %r12;
            sub.u32 %r17, %r10, %r17;
            shl.b32 %r18, %r9, 2;
            cvt.u64.u32 %rd3, %r18;
            mad.wide.u32 %rd3, %r16, %r13, %rd3;
            mad.wide.u32 %rd3, %r17, %r14, %rd3;
            mad.wide.u32 %rd3, %r8, %r15, %rd3;
            shl.b64 %rd3, %rd3, 2;
            add.u64 %rd3, %rd3, %rd1;
            mov.f32 %f0, 0f00000000;
            mov.f32 %f1, 0f00000000;
            mov.f32 %f2, 0f00000000;
            mov.f32 %f3, 0f00000000;
            setp.lt.u32 %p3, %r18, %r2;
            @%p3 ld.global.v4.f32 {%f0, %f1, %f2, %f3}, [%rd3];
            cvt.rn.bf16x2.f32 %r19, %f1, %f0;
            cvt.rn.bf16x2.f32 %r20, %f3, %f2;
            mov.u32 %r21, %nctaid.y;
            mul.wide.u32 %rd5, %r10, %r7;
            cvt.u64.u32 %rd6, %r6;
            add.u64 %rd5, %rd5, %rd6;
            shl.b64 %rd5, %rd5, 3;
            ld.param.u64 %rd7, [p_out];
            cvta.to.global.u64 %rd7, %rd7;
            add.u64 %rd7, %rd7, %rd5;
            mul.wide.u32 %rd8, %r21, %r7;
            shl.b64 %rd8, %rd8, 3;
            @%p2 add.u64 %rd7, %rd7, %rd8;
            st.global.v2.b32 [%rd7], {%r19, %r20};
        DONE:
            ret;
        }

        """);

    // Issues the copies of key tile `tile` (absolute row) into the stage at shared address `stage` (keys, then values a
    // tile further): 16 bytes each, the row's V·2 bytes and then zeros up to D (and zeros at or past the block's last key,
    // %r37), every thread `chunks / threads` copies apart. Expects %rd11 / %rd12 = the head's packed keys / values.
    // Uses %r11-%r19, %rd1-%rd3, %p1, %p21.
    private static string SpanPipelinedIssue(int v, int d, int threads, string tile, string stage)
    {
        int chunks = d / 8, packedChunks = v / 8, total = SpanTensorKeys * chunks, stride = FlashStride(d);
        int tileBytes = SpanTensorKeys * stride;
        var s = new StringBuilder();
        for (int it = 0; it * threads < total; it++)
        {
            bool guard = (it + 1) * threads > total;
            s.AppendLine(CultureInfo.InvariantCulture, $"""
                    add.u32 %r11, %r1, {it * threads};
                    setp.lt.u32 %p21, %r11, {total};
                    div.u32 %r12, %r11, {chunks};
                    rem.u32 %r13, %r11, {chunks};
                    add.u32 %r14, %r12, {tile};
                    setp.lt.u32 %p1, %r14, %r37;
                    setp.lt.and.u32 %p1, %r13, {packedChunks}, %p1;
                    selp.u32 %r15, 16, 0, %p1;
                    selp.u32 %r14, %r14, 0, %p1;
                    selp.u32 %r16, %r13, 0, %p1;
                    mul.lo.u32 %r17, %r14, {2 * v};
                    shl.b32 %r16, %r16, 4;
                    add.u32 %r17, %r17, %r16;
                    cvt.u64.u32 %rd1, %r17;
                    add.u64 %rd2, %rd11, %rd1;
                    add.u64 %rd3, %rd12, %rd1;
                    mad.lo.u32 %r18, %r12, {stride}, {stage};
                    shl.b32 %r19, %r13, 4;
                    add.u32 %r18, %r18, %r19;
                    {(guard ? "@%p21 " : "")}cp.async.cg.shared.global [%r18], [%rd2], 16, %r15;
                    add.u32 %r18, %r18, {tileBytes};
                    {(guard ? "@%p21 " : "")}cp.async.cg.shared.global [%r18], [%rd3], 16, %r15;
                """);
        }

        return s.ToString();
    }

    private static void SpanPipelined(StringBuilder sb, int v, int warps)
    {
        int d = (v + 15) / 16 * 16, stride = FlashStride(d), kSteps = d / 16, vTiles = v / 8, threads = 32 * warps, rowsPerBlock = 16 * warps;
        int tileBytes = SpanTensorKeys * stride, stageBytes = 2 * tileBytes;
        string name = SpanPipelinedName(v, warps);
        var s = new StringBuilder();
        s.AppendLine(CultureInfo.InvariantCulture, $$"""
            .visible .entry {{name}}(
                {{SpanParameters}},
                {{SpanLayoutParameters}}
            )
            .maxntid {{threads}}, 1, 1
            {
            {{FlashRegisters(d, 4 * vTiles)}}
                .reg .b32 %pl<12>;
                .shared .align 4 .b32 {{name}}_range[{{2 * warps}}];
            {{FlashLanes(stride)}}
                ld.param.u64 %rd10, [p_q];
                ld.param.u64 %rd11, [p_k];
                ld.param.u64 %rd12, [p_v];
                ld.param.u64 %rd13, [p_st];
                ld.param.u64 %rd16, [p_en];
                ld.param.u64 %rd14, [p_y];
                ld.param.u64 %rd15, [p_lse];
                cvta.to.global.u64 %rd10, %rd10;
                cvta.to.global.u64 %rd11, %rd11;
                cvta.to.global.u64 %rd12, %rd12;
                cvta.to.global.u64 %rd13, %rd13;
                cvta.to.global.u64 %rd16, %rd16;
                cvta.to.global.u64 %rd14, %rd14;
                ld.param.u32 %r20, [p_rows];
                ld.param.u32 %r21, [p_keys];
                ld.param.u32 %r22, [p_dim];
                ld.param.f32 %f40, [p_scale];
                ld.param.u32 %r23, [p_group];
                ld.param.u32 %r28, [p_hpt];
                ld.param.u32 %pl0, [p_hq];
                ld.param.u32 %pl1, [p_qb];
                ld.param.u32 %pl2, [p_qh];
                ld.param.u32 %pl3, [p_qr];
                ld.param.u32 %pl4, [p_yb];
                ld.param.u32 %pl5, [p_yh];
                ld.param.u32 %pl6, [p_yr];
                mov.u32 %r24, %ctaid.y;
                mov.u32 %r25, %ctaid.x;
                mul.lo.u32 %r25, %r25, {{rowsPerBlock}};
                div.u32 %pl7, %r24, %pl0;
                mul.lo.u32 %pl8, %pl7, %pl0;
                sub.u32 %pl8, %r24, %pl8;
                mul.wide.u32 %rd40, %pl7, %pl1;
                mad.wide.u32 %rd40, %pl8, %pl2, %rd40;
                shl.b64 %rd40, %rd40, 2;
                add.u64 %rd10, %rd10, %rd40;
                mul.wide.u32 %rd40, %pl7, %pl4;
                mad.wide.u32 %rd40, %pl8, %pl5, %rd40;
                shl.b64 %rd40, %rd40, 2;
                add.u64 %rd14, %rd14, %rd40;
                div.u32 %r44, %r24, %r23;
                mul.lo.u32 %r44, %r44, %r21;
                mul.wide.u32 %rd41, %r44, {{2 * v}};
                add.u64 %rd11, %rd11, %rd41;
                add.u64 %rd12, %rd12, %rd41;
                div.u32 %r28, %r24, %r28;
                mul.lo.u32 %r28, %r28, %r20;
                mov.u32 %r26, spans_tcp_d{{v}}_smem;
                add.u32 %r27, %r26, {{stageBytes}};
                mov.u32 %r48, {{name}}_range;
            """);

        // This thread's rows (g and g + 8 of its warp's 16): their ranges; then the block's keys, so the first tile's copies
        // start before the queries are staged.
        s.AppendLine(CultureInfo.InvariantCulture, $"""
                shl.b32 %r30, %r3, 4;
                add.u32 %r30, %r30, %r25;
                add.u32 %r30, %r30, %r4;
                add.u32 %r31, %r30, 8;
            {SpanRange("%r30", "%r32", "%r33", "%r20", "%r21", "%r28", "%rd13", "%rd16", "%r44", "%rd44", "%rd45", "%f61", "%f62", "%p2")}
            {SpanRange("%r31", "%r34", "%r35", "%r20", "%r21", "%r28", "%rd13", "%rd16", "%r44", "%rd44", "%rd45", "%f61", "%f62", "%p2")}
                and.b32 %r45, %r1, {rowsPerBlock - 1};
                add.u32 %r45, %r45, %r25;
            {SpanRange("%r45", "%r46", "%r47", "%r20", "%r21", "%r28", "%rd13", "%rd16", "%r44", "%rd44", "%rd45", "%f61", "%f62", "%p2")}
            """);
        s.AppendLine(SpanBlockRange("%r46", "%r47", "%r21", "%r2", "%r3", warps, "%r48", "%r36", "%r37", "%r44", "%r45", "%p3"));

        // The first tile into stage 0, then the queries through stage 1 into their fragments.
        s.AppendLine("""
                setp.lt.u32 %p14, %r36, %r37;
                @!%p14 bra ISSUED0;
            """);
        s.AppendLine(SpanPipelinedIssue(v, d, threads, "%r36", "%r26"));
        s.AppendLine("""
            ISSUED0:
                cp.async.commit_group;
            """);
        s.AppendLine(SpanTensorStage(rowsPerBlock, d, threads, "%r27", "%r25", "%r20", "%rd10", "QS", rowStride: "%pl3"));
        s.AppendLine(CultureInfo.InvariantCulture, $"""
                bar.sync 0;
                shl.b32 %r29, %r3, 4;
                mul.lo.u32 %r29, %r29, {stride};
                add.u32 %r29, %r29, %r27;
                add.u32 %r29, %r29, %r6;
            """);
        for (int ks = 0; ks < kSteps; ks++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%qa{4 * ks}, %qa{4 * ks + 1}, %qa{4 * ks + 2}, %qa{4 * ks + 3}}}, [%r29+{ks * 32}];");
        }

        // Every warp has its fragments before the loop's first copies overwrite stage 1.
        s.AppendLine($"""
                bar.sync 0;
                mov.f32 %f42, 0fFF800000;
                mov.f32 %f43, 0fFF800000;
                mov.f32 %f44, 0f00000000;
                mov.f32 %f45, 0f00000000;
                shl.b32 %r42, %r5, 1;
            {Zeros("%acc", 4 * vTiles)}
                mov.u32 %r43, %r36;
                mov.u32 %r56, 0;
            TILE:
                setp.ge.u32 %p3, %r43, %r37;
                @%p3 bra TILE_END;
                add.u32 %r57, %r43, {SpanTensorKeys};
                xor.b32 %r58, %r56, {stageBytes};
                add.u32 %r58, %r58, %r26;
                setp.lt.u32 %p14, %r57, %r37;
                @!%p14 bra ISSUED;
            """);
        s.AppendLine(SpanPipelinedIssue(v, d, threads, "%r57", "%r58"));
        s.AppendLine(CultureInfo.InvariantCulture, $"""
            ISSUED:
                cp.async.commit_group;
                cp.async.wait_group 1;
                bar.sync 0;
                add.u32 %r59, %r26, %r56;
                add.u32 %r41, %r59, %r7;
                add.u32 %r40, %r59, {tileBytes};
                add.u32 %r40, %r40, %r6;
            """);
        s.AppendLine(Zeros("%s", 32));
        for (int ks = 0; ks < kSteps; ks++)
        {
            for (int nt2 = 0; nt2 < 4; nt2++)
            {
                s.AppendLine(LdMatrix("%tb", "%r41", nt2 * 16 * stride + ks * 32, transpose: false));
                s.AppendLine(string.Create(CultureInfo.InvariantCulture, $"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {{%s{8 * nt2}, %s{8 * nt2 + 1}, %s{8 * nt2 + 2}, %s{8 * nt2 + 3}}}, ")
                             + string.Create(CultureInfo.InvariantCulture, $"{{%qa{4 * ks}, %qa{4 * ks + 1}, %qa{4 * ks + 2}, %qa{4 * ks + 3}}}, {{%tb0, %tb1}}, {{%s{8 * nt2}, %s{8 * nt2 + 1}, %s{8 * nt2 + 2}, %s{8 * nt2 + 3}}};"));
                s.AppendLine(string.Create(CultureInfo.InvariantCulture, $"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {{%s{8 * nt2 + 4}, %s{8 * nt2 + 5}, %s{8 * nt2 + 6}, %s{8 * nt2 + 7}}}, ")
                             + string.Create(CultureInfo.InvariantCulture, $"{{%qa{4 * ks}, %qa{4 * ks + 1}, %qa{4 * ks + 2}, %qa{4 * ks + 3}}}, {{%tb2, %tb3}}, {{%s{8 * nt2 + 4}, %s{8 * nt2 + 5}, %s{8 * nt2 + 6}, %s{8 * nt2 + 7}}};"));
            }
        }

        // Scale, mask by each row's range (key c = tile + 8 nt + 2t + j; row g: [%r32, %r33), g + 8: [%r34, %r35)), maxima:
        // as the single-stage kernel.
        s.AppendLine("""
                add.u32 %r49, %r43, %r42;
                mov.f32 %f46, 0fFF800000;
                mov.f32 %f47, 0fFF800000;
            """);
        for (int nt = 0; nt < 8; nt++)
        {
            for (int jj = 0; jj < 2; jj++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"""
                        add.u32 %r50, %r49, {nt * 8 + jj};
                        setp.lt.s32 %p4, %r50, %r33;
                        setp.lt.s32 %p5, %r50, %r35;
                        setp.ge.and.s32 %p4, %r50, %r32, %p4;
                        setp.ge.and.s32 %p5, %r50, %r34, %p5;
                        mul.f32 %s{4 * nt + jj}, %s{4 * nt + jj}, %f40;
                        mul.f32 %s{4 * nt + jj + 2}, %s{4 * nt + jj + 2}, %f40;
                        selp.f32 %s{4 * nt + jj}, %s{4 * nt + jj}, 0fFF800000, %p4;
                        selp.f32 %s{4 * nt + jj + 2}, %s{4 * nt + jj + 2}, 0fFF800000, %p5;
                        max.f32 %f46, %f46, %s{4 * nt + jj};
                        max.f32 %f47, %f47, %s{4 * nt + jj + 2};
                    """);
            }
        }

        s.AppendLine("""
                shfl.sync.bfly.b32 %f48, %f46, 1, 31, 0xffffffff;
                max.f32 %f46, %f46, %f48;
                shfl.sync.bfly.b32 %f48, %f46, 2, 31, 0xffffffff;
                max.f32 %f46, %f46, %f48;
                shfl.sync.bfly.b32 %f48, %f47, 1, 31, 0xffffffff;
                max.f32 %f47, %f47, %f48;
                shfl.sync.bfly.b32 %f48, %f47, 2, 31, 0xffffffff;
                max.f32 %f47, %f47, %f48;
                max.f32 %f46, %f46, %f42;
                max.f32 %f47, %f47, %f43;
                setp.eq.f32 %p6, %f46, 0fFF800000;
                selp.f32 %f49, 0f00000000, %f46, %p6;
                setp.eq.f32 %p7, %f47, 0fFF800000;
                selp.f32 %f50, 0f00000000, %f47, %p7;
                sub.f32 %f51, %f42, %f49;
                ex2.approx.ftz.f32 %f51, %f51;
                sub.f32 %f52, %f43, %f50;
                ex2.approx.ftz.f32 %f52, %f52;
                mov.f32 %f42, %f46;
                mov.f32 %f43, %f47;
                mov.f32 %f53, 0f00000000;
                mov.f32 %f54, 0f00000000;
            """);
        for (int nt = 0; nt < 8; nt++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"""
                    sub.f32 %s{4 * nt}, %s{4 * nt}, %f49;
                    ex2.approx.ftz.f32 %s{4 * nt}, %s{4 * nt};
                    sub.f32 %s{4 * nt + 1}, %s{4 * nt + 1}, %f49;
                    ex2.approx.ftz.f32 %s{4 * nt + 1}, %s{4 * nt + 1};
                    sub.f32 %s{4 * nt + 2}, %s{4 * nt + 2}, %f50;
                    ex2.approx.ftz.f32 %s{4 * nt + 2}, %s{4 * nt + 2};
                    sub.f32 %s{4 * nt + 3}, %s{4 * nt + 3}, %f50;
                    ex2.approx.ftz.f32 %s{4 * nt + 3}, %s{4 * nt + 3};
                    add.f32 %f53, %f53, %s{4 * nt};
                    add.f32 %f53, %f53, %s{4 * nt + 1};
                    add.f32 %f54, %f54, %s{4 * nt + 2};
                    add.f32 %f54, %f54, %s{4 * nt + 3};
                """);
        }

        s.AppendLine("""
                fma.rn.f32 %f44, %f44, %f51, %f53;
                fma.rn.f32 %f45, %f45, %f52, %f54;
            """);
        for (int nt = 0; nt < vTiles; nt++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"""
                    mul.f32 %acc{4 * nt}, %acc{4 * nt}, %f51;
                    mul.f32 %acc{4 * nt + 1}, %acc{4 * nt + 1}, %f51;
                    mul.f32 %acc{4 * nt + 2}, %acc{4 * nt + 2}, %f52;
                    mul.f32 %acc{4 * nt + 3}, %acc{4 * nt + 3}, %f52;
                """);
        }

        // O += P·V over the head size's tiles of 8 columns: pairs from one ldmatrix.x4, an odd last tile from an .x2.
        for (int jj = 0; jj < 4; jj++)
        {
            s.AppendLine(PackA("%s", jj));
            for (int dn2 = 0; dn2 < vTiles / 2; dn2++)
            {
                s.AppendLine(LdMatrix("%tb", "%r40", jj * 16 * stride + dn2 * 32, transpose: true));
                s.AppendLine(Mma("%acc", 2 * dn2, "%pa", "%tb0", "%tb1"));
                s.AppendLine(Mma("%acc", 2 * dn2 + 1, "%pa", "%tb2", "%tb3"));
            }

            if (vTiles % 2 == 1)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {{%tb0, %tb1}}, [%r40+{jj * 16 * stride + vTiles / 2 * 32}];");
                s.AppendLine(Mma("%acc", vTiles - 1, "%pa", "%tb0", "%tb1"));
            }
        }

        s.AppendLine(CultureInfo.InvariantCulture, $"""
                bar.sync 0;
                xor.b32 %r56, %r56, {stageBytes};
                add.u32 %r43, %r43, {SpanTensorKeys};
                bra TILE;
            TILE_END:
                shfl.sync.bfly.b32 %f48, %f44, 1, 31, 0xffffffff;
                add.f32 %f44, %f44, %f48;
                shfl.sync.bfly.b32 %f48, %f44, 2, 31, 0xffffffff;
                add.f32 %f44, %f44, %f48;
                shfl.sync.bfly.b32 %f48, %f45, 1, 31, 0xffffffff;
                add.f32 %f45, %f45, %f48;
                shfl.sync.bfly.b32 %f48, %f45, 2, 31, 0xffffffff;
                add.f32 %f45, %f45, %f48;
                setp.gt.f32 %p8, %f44, 0f00000000;
                rcp.rn.f32 %f55, %f44;
                selp.f32 %f55, %f55, 0f00000000, %p8;
                setp.gt.f32 %p9, %f45, 0f00000000;
                rcp.rn.f32 %f56, %f45;
                selp.f32 %f56, %f56, 0f00000000, %p9;
                setp.lt.u32 %p10, %r30, %r20;
                setp.lt.u32 %p11, %r31, %r20;
                mad.lo.u32 %r51, %r30, %pl6, %r42;
                mul.wide.u32 %rd22, %r51, 4;
                add.u64 %rd22, %rd22, %rd14;
                mad.lo.u32 %r51, %r31, %pl6, %r42;
                mul.wide.u32 %rd26, %r51, 4;
                add.u64 %rd26, %rd26, %rd14;
            """);
        for (int nt = 0; nt < vTiles; nt++)
        {
            // Columns 8 nt + 2t and the next: written when inside the head size (a multiple of 4, so both or neither).
            s.AppendLine(CultureInfo.InvariantCulture, $$"""
                    add.u32 %r52, %r42, {{nt * 8}};
                    setp.lt.u32 %p12, %r52, %r22;
                    and.pred %p13, %p12, %p10;
                    and.pred %p12, %p12, %p11;
                    mul.f32 %f57, %acc{{4 * nt}}, %f55;
                    mul.f32 %f58, %acc{{4 * nt + 1}}, %f55;
                    @%p13 st.global.v2.f32 [%rd22+{{nt * 32}}], {%f57, %f58};
                    mul.f32 %f57, %acc{{4 * nt + 2}}, %f56;
                    mul.f32 %f58, %acc{{4 * nt + 3}}, %f56;
                    @%p12 st.global.v2.f32 [%rd26+{{nt * 32}}], {%f57, %f58};
                """);
        }

        // log-sum-exp of the scaled scores (natural log): (m + log2 l) · ln 2, -∞ for a row that saw nothing.
        s.AppendLine("""
                setp.eq.u64 %p12, %rd15, 0;
                @%p12 bra DONE;
                cvta.to.global.u64 %rd15, %rd15;
                setp.ne.u32 %p13, %r5, 0;
                @%p13 bra DONE;
                mul.wide.u32 %rd24, %r24, %r20;
                cvt.u64.u32 %rd25, %r30;
                add.u64 %rd24, %rd24, %rd25;
                shl.b64 %rd24, %rd24, 2;
                add.u64 %rd24, %rd24, %rd15;
                lg2.approx.f32 %f59, %f44;
                add.f32 %f59, %f59, %f42;
                mul.f32 %f59, %f59, 0f3F317218;
                selp.f32 %f59, %f59, 0fFF800000, %p8;
                lg2.approx.f32 %f60, %f45;
                add.f32 %f60, %f60, %f43;
                mul.f32 %f60, %f60, 0f3F317218;
                selp.f32 %f60, %f60, 0fFF800000, %p9;
                @%p10 st.global.f32 [%rd24], %f59;
                @%p11 st.global.f32 [%rd24+32], %f60;
            DONE:
                ret;
            }
            """);
        sb.Append(s);
        sb.AppendLine();
    }
}
