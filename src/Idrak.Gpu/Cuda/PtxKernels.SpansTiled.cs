// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;

namespace Idrak.Gpu.Cuda;

// Attention over one range of keys per query row (Backend.AttentionSpans), the fast kernels: one module per padded head
// size, built and loaded the first time a head size needs it (CudaBackend.Spans.cs), so a GPU loads only the sizes it
// runs.
//
// - attention_spans_f32_d{D} (float32, PTX 6.0 for sm_50 and newer): a block of 128 threads takes 64 query rows of one
//   head (32 for head sizes over 128, so its tiles fit a block: about 69 KB at 256) and walks the keys in tiles of 32.
//   The products are tiled in registers (FlashAttention's outer-product form): thread (tr, tc) = (tid / 8, tid % 8)
//   holds the scores of rows rt·tr .. rt·tr + rt - 1 (rt = 4, or 2 over 128) and keys 4tc .. 4tc + 3 (vector reads of
//   shared memory for every 4·rt multiply-adds), then the outputs of the same rows and the columns tc + 8j.
//   Shared memory (dynamic): Qᵀ [D][rows] (scaled by `scale` as it is staged), Kᵀ [D][32] (the same space then holds V
//   [32][D]), P [rows][36] and the block's key range. The old attention_spans_f32 (one warp per 8 rows, a shared-memory
//   read per multiply-add: four times slower than the composed path where it was measured, plans/done/11-vision-language.md)
//   stays for head sizes this one does not take (not a multiple of 4) and devices whose shared memory per block cannot
//   hold this one's.
//
// Query row i of head h sees keys st[t·rows + i] ≤ c < en[t·rows + i] (clamped to [0, keyRows]; floats holding
// integers), t = h / headsPerTable, key/value head h / group. The block walks the keys from the smallest start of its
// rows (those that see any key) to their largest end; a key outside a
// row's range scores -∞ for it, and a row whose range
// is empty keeps a sum of 0: zeros, and a log-sum-exp of -∞. Head sizes that are multiples of 4 (rows read as float4):
// up to 256. Grid: x = ⌈rows / rows per block⌉, y = heads.
internal static partial class PtxKernels
{
    /// <summary>The largest head size of the float32 span kernel (A1: 129 to 256 in blocks of 32 rows, 2 a thread).</summary>
    public const int SpanMaxDim = 256;

    /// <summary>
    /// Query rows per block of the float32 span kernel for padded head size <paramref name="d"/>: 64 (4 a thread) up to
    /// 128; 32 (2 a thread) above, so Qᵀ, Kᵀ and P stay within what a block may hold (about 69 KB at 256).
    /// </summary>
    public static int SpanFloatRows(int d) => 16 * SpanRowsPerThread(d);

    // Query rows each thread of the float32 span kernel holds (its 4 key columns each).
    private static int SpanRowsPerThread(int d) => d <= FlashMaxDim ? 4 : 2;

    /// <summary>Key positions per tile of the float32 span kernel.</summary>
    public const int SpanFloatKeys = 32;

    // Row stride of the float32 kernel's P tile (floats): a multiple of 4 for its vector reads and writes.
    private const int SpanPStride = SpanFloatKeys + 4;

    /// <summary>
    /// The padded head size of the float32 span kernel for <paramref name="dim"/> (a multiple of 8 up to 128, of 16 above, up to
    /// <see cref="SpanMaxDim"/>), or 0 when it does not take it.
    /// </summary>
    public static int SpanFloatDim(int dim) => dim > 0 && dim % 4 == 0 && dim <= SpanMaxDim ? dim <= FlashMaxDim ? (dim + 7) / 8 * 8 : (dim + 15) / 16 * 16 : 0;

    /// <summary>The float32 span kernel's name for padded head size <paramref name="d"/>.</summary>
    public static string SpanFloatName(int d) => string.Create(CultureInfo.InvariantCulture, $"attention_spans_f32_d{d}");

    /// <summary>Dynamic shared memory of the float32 span kernel (bytes): Qᵀ, Kᵀ (then V), P and the block's range.</summary>
    public static int SpanFloatShared(int d) => 4 * (SpanFloatRows(d) * d + SpanFloatKeys * d + SpanFloatRows(d) * SpanPStride + 8);

    /// <summary>The float32 span module for padded head size <paramref name="d"/> (PTX 6.0, sm_50, as the main module).</summary>
    public static (string Name, string[] Kernels, string Source) SpanFloatModule(int d) =>
        Module(string.Create(CultureInfo.InvariantCulture, $"attention spans f32 d{d}"), sb =>
        {
            SpanFloat(sb, d, strided: false);
            SpanFloat(sb, d, strided: true);
        }, version: "6.0", target: "sm_50");

    /// <summary>The float32 span kernel's name for padded head size <paramref name="d"/> reading rows through a layout (AttentionSpansStrided).</summary>
    public static string SpanFloatStridedName(int d) => string.Create(CultureInfo.InvariantCulture, $"attention_spans_f32s_d{d}");

    // The row layout of queries and outputs of the strided float32 kernel (after SpanParameters).
    private const string SpanLayoutParameters =
        ".param .u32 p_hq, .param .u32 p_qb, .param .u32 p_qh, .param .u32 p_qr, .param .u32 p_yb, .param .u32 p_yh, .param .u32 p_yr";

    // The row layout of keys and values of the strided float32 kernel (after SpanLayoutParameters): key/value head
    // g = b·p_hk + h, key c at k (or v) + b·p_kb + h·p_kh + c·p_kr (floats).
    private const string SpanKeyLayoutParameters = ".param .u32 p_hk, .param .u32 p_kb, .param .u32 p_kh, .param .u32 p_kr";

    // A module of its own: its PTX version and target, then the kernels `build` writes.
    private static (string Name, string[] Kernels, string Source) Module(string name, Action<StringBuilder> build, string version, string target)
    {
        var sb = new StringBuilder();
        sb.AppendLine($".version {version}");
        sb.AppendLine($".target {target}");
        sb.AppendLine(".address_size 64");
        sb.AppendLine();
        build(sb);
        string source = sb.ToString();
        return (name, KernelNames(source), source);
    }

    // The parameters of the kernels (as attention_spans_f32's).
    private const string SpanParameters =
        ".param .u64 p_q, .param .u64 p_k, .param .u64 p_v, .param .u64 p_st, .param .u64 p_en, .param .u64 p_y, .param .u64 p_lse, "
        + ".param .u32 p_rows, .param .u32 p_keys, .param .u32 p_dim, .param .f32 p_scale, .param .u32 p_group, .param .u32 p_hpt";

    // lo, hi = the range of table row `row` clamped to [0, keyRows] (0, 0 past the head's rows). Expects rows, keyRows,
    // the table's first row and the starts / ends pointers in the registers named; uses `t`, `a`, `b`, `fa`, `fb`, `p`.
    private static string SpanRange(string row, string lo, string hi, string rows, string keys, string table, string starts, string ends,
        string t, string a, string b, string fa, string fb, string p) => $"""
            mov.u32 {lo}, 0;
            mov.u32 {hi}, 0;
            setp.lt.u32 {p}, {row}, {rows};
            add.u32 {t}, {table}, {row};
            mul.wide.u32 {a}, {t}, 4;
            add.u64 {b}, {a}, {ends};
            add.u64 {a}, {a}, {starts};
            @{p} ld.global.f32 {fa}, [{a}];
            @{p} ld.global.f32 {fb}, [{b}];
            @{p} cvt.rzi.s32.f32 {lo}, {fa};
            @{p} cvt.rzi.s32.f32 {hi}, {fb};
            max.s32 {lo}, {lo}, 0;
            min.s32 {lo}, {lo}, {keys};
            max.s32 {hi}, {hi}, 0;
            min.s32 {hi}, {hi}, {keys};
        """;

    // The block's keys: from = the smallest start of the non-empty ranges among lo, hi of every thread (an empty range
    // counts as [keyRows, 0)), to = their largest end; reduced over the warp, then over the warps through `shared`
    // (2 · warps words). Leaves from, to in the registers named after a barrier. Uses t, u, p.
    private static string SpanBlockRange(string lo, string hi, string keys, string lane, string warp, int warps, string shared,
        string from, string to, string t, string u, string p)
    {
        var s = new StringBuilder();
        s.AppendLine($"""
                setp.gt.s32 {p}, {hi}, {lo};
                selp.b32 {from}, {lo}, {keys}, {p};
                selp.b32 {to}, {hi}, 0, {p};
            """);
        foreach (int offset in new[] { 16, 8, 4, 2, 1 })
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    shfl.sync.bfly.b32 {t}, {from}, {offset}, 31, 0xffffffff;");
            s.AppendLine($"    min.u32 {from}, {from}, {t};");
            s.AppendLine(CultureInfo.InvariantCulture, $"    shfl.sync.bfly.b32 {t}, {to}, {offset}, 31, 0xffffffff;");
            s.AppendLine($"    max.u32 {to}, {to}, {t};");
        }

        s.AppendLine(CultureInfo.InvariantCulture, $"""
                setp.ne.u32 {p}, {lane}, 0;
                shl.b32 {t}, {warp}, 2;
                add.u32 {t}, {t}, {shared};
                @!{p} st.shared.u32 [{t}], {from};
                @!{p} st.shared.u32 [{t}+{4 * warps}], {to};
                bar.sync 0;
                ld.shared.u32 {from}, [{shared}];
                ld.shared.u32 {to}, [{shared}+{4 * warps}];
            """);
        for (int w = 1; w < warps; w++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"""
                    ld.shared.u32 {u}, [{shared}+{4 * w}];
                    min.u32 {from}, {from}, {u};
                    ld.shared.u32 {u}, [{shared}+{4 * (warps + w)}];
                    max.u32 {to}, {to}, {u};
                """);
        }

        return s.ToString();
    }

    // ------------------------------------------------------------------ float32

    // The contiguous kernel's offsets of head %r10 (in the floats of q / y: %r11·dim; of k / v: (%r10 / group)·keys·dim)
    // and the strided kernel's (through the layouts: b = hh / p_hq, h = hh - b·p_hq; g = hh / group, then b, h of g by
    // p_hk), each continuing the line before it; the strided kernel's layout parameters likewise.
    private const string ContiguousOffsets = """

                    mul.wide.u32 %rd8, %r11, %r3;
                    shl.b64 %rd8, %rd8, 2;
                    add.u64 %rd1, %rd1, %rd8;
                    add.u64 %rd6, %rd6, %rd8;
                    div.u32 %r12, %r10, %r4;
                    mul.lo.u32 %r12, %r12, %r2;
                    mul.wide.u32 %rd8, %r12, %r3;
                    shl.b64 %rd8, %rd8, 2;
                    add.u64 %rd2, %rd2, %rd8;
                    add.u64 %rd3, %rd3, %rd8;
                """;

    private const string StridedOffsets = """

                    div.u32 %r61, %r10, %r50;
                    mul.lo.u32 %r62, %r61, %r50;
                    sub.u32 %r62, %r10, %r62;
                    mul.wide.u32 %rd8, %r61, %r51;
                    mad.wide.u32 %rd8, %r62, %r52, %rd8;
                    shl.b64 %rd8, %rd8, 2;
                    add.u64 %rd1, %rd1, %rd8;
                    mul.wide.u32 %rd8, %r61, %r54;
                    mad.wide.u32 %rd8, %r62, %r55, %rd8;
                    shl.b64 %rd8, %rd8, 2;
                    add.u64 %rd6, %rd6, %rd8;
                    div.u32 %r12, %r10, %r4;
                    div.u32 %r61, %r12, %r57;
                    mul.lo.u32 %r62, %r61, %r57;
                    sub.u32 %r62, %r12, %r62;
                    mul.wide.u32 %rd8, %r61, %r58;
                    mad.wide.u32 %rd8, %r62, %r59, %rd8;
                    shl.b64 %rd8, %rd8, 2;
                    add.u64 %rd2, %rd2, %rd8;
                    add.u64 %rd3, %rd3, %rd8;
                """;

    private const string StridedLoads = """

                    ld.param.u32 %r50, [p_hq];
                    ld.param.u32 %r51, [p_qb];
                    ld.param.u32 %r52, [p_qh];
                    ld.param.u32 %r53, [p_qr];
                    ld.param.u32 %r54, [p_yb];
                    ld.param.u32 %r55, [p_yh];
                    ld.param.u32 %r56, [p_yr];
                    ld.param.u32 %r57, [p_hk];
                    ld.param.u32 %r58, [p_kb];
                    ld.param.u32 %r59, [p_kh];
                    ld.param.u32 %r60, [p_kr];
                """;

    // strided: the kernel attention_spans_f32s_d{D}, reading queries, keys and values and writing the output through row
    // layouts (SpanLayoutParameters, SpanKeyLayoutParameters; the log-sum-exp stays [heads, rows]): the same arithmetic in
    // the same order as the contiguous kernel, only the addresses differ.
    private static void SpanFloat(StringBuilder sb, int d, bool strided)
    {
        string name = strided ? SpanFloatStridedName(d) : SpanFloatName(d);
        string qRow = strided ? "%r53" : "%r3", kRow = strided ? "%r60" : "%r3", yRow = strided ? "%r56" : "%r3";
        int j = d / 8;                                 // output columns per thread: tc + 8·jj
        int rt = SpanRowsPerThread(d), rows = 16 * rt; // query rows per thread and per block
        int rtShift = rt == 4 ? 2 : 1, rowShift = rows == 64 ? 6 : 5;
        int qBytes = 4 * rows * d, kBytes = 4 * SpanFloatKeys * d, pBytes = 4 * rows * SpanPStride;
        int quads = d / 4;
        var s = new StringBuilder();
        s.AppendLine(CultureInfo.InvariantCulture, $$"""
            .extern .shared .align 16 .b8 {{name}}_smem[];
            .visible .entry {{name}}(
                {{SpanParameters}}, .param .f32 p_softcap{{(strided ? ",\n    " + SpanLayoutParameters + ",\n    " + SpanKeyLayoutParameters : "")}}
            )
            {
                .reg .pred %p<16>;
                .reg .b32 %r<64>;
                .reg .b64 %rd<24>;
                .reg .f32 %f<24>;
                .reg .f32 %s<{{4 * rt}}>;
                .reg .f32 %o<{{rt * j}}>;
                .reg .f32 %m<{{rt}}>;
                .reg .f32 %l<{{rt}}>;
                .reg .b32 %lo<{{rt}}>;
                .reg .b32 %hi<{{rt}}>;
                .reg .f32 %qv<{{rt}}>;
                .reg .f32 %kv<4>;
                .reg .f32 %softcap, %captmp;
                .reg .pred %capped;
                ld.param.u64 %rd1, [p_q];
                ld.param.u64 %rd2, [p_k];
                ld.param.u64 %rd3, [p_v];
                ld.param.u64 %rd4, [p_st];
                ld.param.u64 %rd5, [p_en];
                ld.param.u64 %rd6, [p_y];
                ld.param.u64 %rd7, [p_lse];
                cvta.to.global.u64 %rd1, %rd1;
                cvta.to.global.u64 %rd2, %rd2;
                cvta.to.global.u64 %rd3, %rd3;
                cvta.to.global.u64 %rd4, %rd4;
                cvta.to.global.u64 %rd5, %rd5;
                cvta.to.global.u64 %rd6, %rd6;
                setp.ne.u64 %p15, %rd7, 0;
                @%p15 cvta.to.global.u64 %rd7, %rd7;
                ld.param.u32 %r1, [p_rows];
                ld.param.u32 %r2, [p_keys];
                ld.param.u32 %r3, [p_dim];
                ld.param.f32 %f1, [p_scale];
                ld.param.u32 %r4, [p_group];
                ld.param.u32 %r5, [p_hpt];
                ld.param.f32 %softcap, [p_softcap];{{(strided ? StridedLoads : "")}}
                setp.gt.f32 %capped, %softcap, 0f00000000;
                mov.u32 %r6, %tid.x;
                and.b32 %r7, %r6, 7;
                shr.u32 %r8, %r6, 3;
                mov.u32 %r9, %ctaid.x;
                shl.b32 %r9, %r9, {{rowShift}};
                mov.u32 %r10, %ctaid.y;
                mul.lo.u32 %r11, %r10, %r1;{{(strided ? StridedOffsets : ContiguousOffsets)}}
                div.u32 %r13, %r10, %r5;
                mul.lo.u32 %r13, %r13, %r1;
                mov.u32 %r14, {{name}}_smem;
                add.u32 %r15, %r14, {{qBytes}};
                add.u32 %r16, %r15, {{kBytes}};
                add.u32 %r17, %r16, {{pBytes}};
            """);

        // Each thread's rows rt·tr + i: their ranges, running maxima and sums.
        for (int i = 0; i < rt; i++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"""
                    shl.b32 %r19, %r8, {rtShift};
                    add.u32 %r19, %r19, %r9;
                    add.u32 %r19, %r19, {i};
                {SpanRange("%r19", string.Create(CultureInfo.InvariantCulture, $"%lo{i}"), string.Create(CultureInfo.InvariantCulture, $"%hi{i}"), "%r1", "%r2", "%r13", "%rd4", "%rd5", "%r18", "%rd9", "%rd10", "%f2", "%f3", "%p1")}
                    mov.f32 %m{i}, 0fFF800000;
                    mov.f32 %l{i}, 0f00000000;
                """);
        }

        for (int i = 0; i < rt * j; i++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    mov.f32 %o{i}, 0f00000000;");
        }

        // The block's keys (thread t reads row t % rows' range).
        s.AppendLine(CultureInfo.InvariantCulture, $"""
                and.b32 %r19, %r6, {rows - 1};
                add.u32 %r19, %r19, %r9;
            {SpanRange("%r19", "%r22", "%r23", "%r1", "%r2", "%r13", "%rd4", "%rd5", "%r18", "%rd9", "%rd10", "%f2", "%f3", "%p1")}
                and.b32 %r24, %r6, 31;
                shr.u32 %r25, %r6, 5;
            """);

        // Qᵀ[d][r] = scale · q[row0 + r, d], zeros past the rows and the head size: thread t takes row t % rows of the
        // quads t / rows, + 128 / rows, ... (a warp writes 32 consecutive floats of a Qᵀ row).
        for (int it = 0; it < rows * d / 512; it++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $$"""
                    add.u32 %r26, %r6, {{it * 128}};
                    and.b32 %r27, %r26, {{rows - 1}};
                    shr.u32 %r28, %r26, {{rowShift}};
                    shl.b32 %r28, %r28, 2;
                    add.u32 %r29, %r9, %r27;
                    setp.lt.u32 %p1, %r29, %r1;
                    setp.lt.and.u32 %p1, %r28, %r3, %p1;
                    cvt.u64.u32 %rd9, %r28;
                    mad.wide.u32 %rd9, %r29, {{qRow}}, %rd9;
                    shl.b64 %rd9, %rd9, 2;
                    add.u64 %rd9, %rd9, %rd1;
                    mov.f32 %f4, 0f00000000;
                    mov.f32 %f5, 0f00000000;
                    mov.f32 %f6, 0f00000000;
                    mov.f32 %f7, 0f00000000;
                    @%p1 ld.global.v4.f32 {%f4, %f5, %f6, %f7}, [%rd9];
                    mul.f32 %f4, %f4, %f1;
                    mul.f32 %f5, %f5, %f1;
                    mul.f32 %f6, %f6, %f1;
                    mul.f32 %f7, %f7, %f1;
                    mad.lo.u32 %r30, %r28, {{rows}}, %r27;
                    shl.b32 %r30, %r30, 2;
                    add.u32 %r30, %r30, %r14;
                    st.shared.f32 [%r30], %f4;
                    st.shared.f32 [%r30+{{4 * rows}}], %f5;
                    st.shared.f32 [%r30+{{8 * rows}}], %f6;
                    st.shared.f32 [%r30+{{12 * rows}}], %f7;
                """);
        }

        s.AppendLine(SpanBlockRange("%r22", "%r23", "%r2", "%r24", "%r25", 4, "%r17", "%r20", "%r21", "%r26", "%r27", "%p2"));

        // The tiles: Kᵀ, scores, the online softmax, P, V, O += P·V.
        int kElements = SpanFloatKeys * quads, kIterations = (kElements + 127) / 128;
        s.AppendLine("""
                mov.u32 %r31, %r20;
            TILE:
                setp.ge.u32 %p3, %r31, %r21;
                @%p3 bra TILE_END;
            """);
        for (int it = 0; it < kIterations; it++)
        {
            // Kᵀ[d][c] = k[tile + c, d]: thread t takes key t % 32 of the quads t / 32, +4, ...
            s.AppendLine(CultureInfo.InvariantCulture, $$"""
                    add.u32 %r26, %r6, {{it * 128}};
                    setp.lt.u32 %p4, %r26, {{kElements}};
                    and.b32 %r27, %r26, 31;
                    shr.u32 %r28, %r26, 5;
                    shl.b32 %r28, %r28, 2;
                    add.u32 %r29, %r31, %r27;
                    setp.lt.and.u32 %p1, %r29, %r21, %p4;
                    setp.lt.and.u32 %p1, %r28, %r3, %p1;
                    cvt.u64.u32 %rd9, %r28;
                    mad.wide.u32 %rd9, %r29, {{kRow}}, %rd9;
                    shl.b64 %rd9, %rd9, 2;
                    add.u64 %rd9, %rd9, %rd2;
                    mov.f32 %f4, 0f00000000;
                    mov.f32 %f5, 0f00000000;
                    mov.f32 %f6, 0f00000000;
                    mov.f32 %f7, 0f00000000;
                    @%p1 ld.global.v4.f32 {%f4, %f5, %f6, %f7}, [%rd9];
                    mad.lo.u32 %r30, %r28, {{SpanFloatKeys}}, %r27;
                    shl.b32 %r30, %r30, 2;
                    add.u32 %r30, %r30, %r15;
                    @%p4 st.shared.f32 [%r30], %f4;
                    @%p4 st.shared.f32 [%r30+{{4 * SpanFloatKeys}}], %f5;
                    @%p4 st.shared.f32 [%r30+{{8 * SpanFloatKeys}}], %f6;
                    @%p4 st.shared.f32 [%r30+{{12 * SpanFloatKeys}}], %f7;
                """);
        }

        s.AppendLine("bar.sync 0;");
        for (int i = 0; i < 4 * rt; i++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    mov.f32 %s{i}, 0f00000000;");
        }

        // S[rt·tr + i, 4tc + jj] = Σ_d Qᵀ[d][rt·tr + i] · Kᵀ[d][4tc + jj], eight d per loop step.
        s.AppendLine(CultureInfo.InvariantCulture, $"""
                shl.b32 %r32, %r8, {rtShift + 2};
                add.u32 %r32, %r32, %r14;
                shl.b32 %r33, %r7, 4;
                add.u32 %r33, %r33, %r15;
                mov.u32 %r34, 0;
            DOT:
                setp.ge.u32 %p5, %r34, {d / 8};
                @%p5 bra DOT_END;
            """);
        for (int u = 0; u < 8; u++)
        {
            if (rt == 4)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    ld.shared.v4.f32 {{%qv0, %qv1, %qv2, %qv3}}, [%r32+{u * 4 * rows}];");
            }
            else
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    ld.shared.v2.f32 {{%qv0, %qv1}}, [%r32+{u * 4 * rows}];");
            }

            s.AppendLine(CultureInfo.InvariantCulture, $"    ld.shared.v4.f32 {{%kv0, %kv1, %kv2, %kv3}}, [%r33+{u * 4 * SpanFloatKeys}];");
            for (int i = 0; i < rt; i++)
            {
                for (int c = 0; c < 4; c++)
                {
                    s.AppendLine(CultureInfo.InvariantCulture, $"    fma.rn.f32 %s{4 * i + c}, %qv{i}, %kv{c}, %s{4 * i + c};");
                }
            }
        }

        s.AppendLine(CultureInfo.InvariantCulture, $"""
                add.u32 %r32, %r32, {8 * 4 * rows};
                add.u32 %r33, %r33, {8 * 4 * SpanFloatKeys};
                add.u32 %r34, %r34, 1;
                bra DOT;
            DOT_END:
                bar.sync 0;
            """);

        // V[c][d] = v[tile + c, d] over Kᵀ's space (quads along the row: coalesced reads, consecutive writes).
        int vIterations = (kElements + 127) / 128;
        for (int it = 0; it < vIterations; it++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $$"""
                    add.u32 %r26, %r6, {{it * 128}};
                    setp.lt.u32 %p4, %r26, {{kElements}};
                    div.u32 %r27, %r26, {{quads}};
                    rem.u32 %r28, %r26, {{quads}};
                    shl.b32 %r28, %r28, 2;
                    add.u32 %r29, %r31, %r27;
                    setp.lt.and.u32 %p1, %r29, %r21, %p4;
                    setp.lt.and.u32 %p1, %r28, %r3, %p1;
                    cvt.u64.u32 %rd9, %r28;
                    mad.wide.u32 %rd9, %r29, {{kRow}}, %rd9;
                    shl.b64 %rd9, %rd9, 2;
                    add.u64 %rd9, %rd9, %rd3;
                    mov.f32 %f4, 0f00000000;
                    mov.f32 %f5, 0f00000000;
                    mov.f32 %f6, 0f00000000;
                    mov.f32 %f7, 0f00000000;
                    @%p1 ld.global.v4.f32 {%f4, %f5, %f6, %f7}, [%rd9];
                    mad.lo.u32 %r30, %r27, {{d}}, %r28;
                    shl.b32 %r30, %r30, 2;
                    add.u32 %r30, %r30, %r15;
                    @%p4 st.shared.v4.f32 [%r30], {%f4, %f5, %f6, %f7};
                """);
        }

        // Softmax per row (log2 domain): soft-cap, mask, the row's maximum over the 8 threads of the row, the rescale, P.
        s.AppendLine("""
                shl.b32 %r35, %r7, 2;
                add.u32 %r35, %r35, %r31;
            """);
        for (int i = 0; i < rt; i++)
        {
            s.AppendLine("    mov.f32 %f8, 0fFF800000;");
            for (int c = 0; c < 4; c++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"""
                        {SoftcapPtx(string.Create(CultureInfo.InvariantCulture, $"%s{4 * i + c}"), "%softcap", "%capped", "%captmp")}
                        mul.f32 %s{4 * i + c}, %s{4 * i + c}, 0f3FB8AA3B;
                        add.u32 %r36, %r35, {c};
                        setp.lt.s32 %p6, %r36, %hi{i};
                        setp.ge.and.s32 %p6, %r36, %lo{i}, %p6;
                        selp.f32 %s{4 * i + c}, %s{4 * i + c}, 0fFF800000, %p6;
                        max.f32 %f8, %f8, %s{4 * i + c};
                    """);
            }

            foreach (int offset in new[] { 1, 2, 4 })
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    shfl.sync.bfly.b32 %f9, %f8, {offset}, 31, 0xffffffff;");
                s.AppendLine("    max.f32 %f8, %f8, %f9;");
            }

            s.AppendLine(CultureInfo.InvariantCulture, $"""
                    max.f32 %f8, %f8, %m{i};
                    setp.eq.f32 %p7, %f8, 0fFF800000;
                    selp.f32 %f10, 0f00000000, %f8, %p7;
                    sub.f32 %f11, %m{i}, %f10;
                    ex2.approx.ftz.f32 %f11, %f11;
                    mov.f32 %m{i}, %f8;
                    mul.f32 %l{i}, %l{i}, %f11;
                """);
            for (int c = 0; c < 4; c++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"""
                        sub.f32 %s{4 * i + c}, %s{4 * i + c}, %f10;
                        ex2.approx.ftz.f32 %s{4 * i + c}, %s{4 * i + c};
                        add.f32 %l{i}, %l{i}, %s{4 * i + c};
                    """);
            }

            for (int c = 0; c < j; c++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    mul.f32 %o{i * j + c}, %o{i * j + c}, %f11;");
            }

            // P[rt·tr + i][4tc .. 4tc + 3]
            s.AppendLine(CultureInfo.InvariantCulture, $$"""
                    shl.b32 %r37, %r8, {{rtShift}};
                    add.u32 %r37, %r37, {{i}};
                    mul.lo.u32 %r37, %r37, {{SpanPStride}};
                    add.u32 %r37, %r37, %r35;
                    sub.u32 %r37, %r37, %r31;
                    shl.b32 %r37, %r37, 2;
                    add.u32 %r37, %r37, %r16;
                    st.shared.v4.f32 [%r37], {%s{{4 * i}}, %s{{4 * i + 1}}, %s{{4 * i + 2}}, %s{{4 * i + 3}}};
                """);
        }

        // O[rt·tr + i, tc + 8c] += Σ_key P[rt·tr + i, key] · V[key, tc + 8c], four keys per loop step (P read as float4).
        s.AppendLine(CultureInfo.InvariantCulture, $"""
                bar.sync 0;
                shl.b32 %r38, %r8, {rtShift};
                mul.lo.u32 %r38, %r38, {4 * SpanPStride};
                add.u32 %r38, %r38, %r16;
                shl.b32 %r39, %r7, 2;
                add.u32 %r39, %r39, %r15;
                mov.u32 %r34, 0;
            PV:
                setp.ge.u32 %p5, %r34, {SpanFloatKeys / 4};
                @%p5 bra PV_END;
            """);
        for (int i = 0; i < rt; i++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    ld.shared.v4.f32 {{%s{4 * i}, %s{4 * i + 1}, %s{4 * i + 2}, %s{4 * i + 3}}}, [%r38+{i * 4 * SpanPStride}];");
        }

        for (int u = 0; u < 4; u++)
        {
            for (int c = 0; c < j; c++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    ld.shared.f32 %f12, [%r39+{u * 4 * d + c * 32}];");
                for (int i = 0; i < rt; i++)
                {
                    s.AppendLine(CultureInfo.InvariantCulture, $"    fma.rn.f32 %o{i * j + c}, %s{4 * i + u}, %f12, %o{i * j + c};");
                }
            }
        }

        s.AppendLine(CultureInfo.InvariantCulture, $"""
                add.u32 %r38, %r38, 16;
                add.u32 %r39, %r39, {4 * 4 * d};
                add.u32 %r34, %r34, 1;
                bra PV;
            PV_END:
                bar.sync 0;
                add.u32 %r31, %r31, {SpanFloatKeys};
                bra TILE;
            TILE_END:
            """);

        // o /= l (the row's sum over its 8 threads; zeros when it saw nothing), lse = (m + log2 l) · ln 2 (-∞ then).
        for (int i = 0; i < rt; i++)
        {
            foreach (int offset in new[] { 1, 2, 4 })
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    shfl.sync.bfly.b32 %f13, %l{i}, {offset}, 31, 0xffffffff;");
                s.AppendLine(CultureInfo.InvariantCulture, $"    add.f32 %l{i}, %l{i}, %f13;");
            }

            s.AppendLine(CultureInfo.InvariantCulture, $"""
                    shl.b32 %r40, %r8, {rtShift};
                    add.u32 %r40, %r40, %r9;
                    add.u32 %r40, %r40, {i};
                    setp.ge.u32 %p9, %r40, %r1;
                    @%p9 bra WRITTEN{i};
                    setp.gt.f32 %p11, %l{i}, 0f00000000;
                    rcp.rn.f32 %f14, %l{i};
                    selp.f32 %f14, %f14, 0f00000000, %p11;
                    mad.lo.u32 %r41, %r40, {yRow}, %r7;
                    mul.wide.u32 %rd11, %r41, 4;
                    add.u64 %rd11, %rd11, %rd6;
                """);
            for (int c = 0; c < j; c++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"""
                        add.u32 %r42, %r7, {8 * c};
                        setp.lt.u32 %p12, %r42, %r3;
                        mul.f32 %f15, %o{i * j + c}, %f14;
                        @%p12 st.global.f32 [%rd11+{32 * c}], %f15;
                    """);
            }

            s.AppendLine(CultureInfo.InvariantCulture, $"""
                    setp.eq.u32 %p10, %r7, 0;
                    and.pred %p10, %p10, %p15;
                    @!%p10 bra WRITTEN{i};
                    lg2.approx.ftz.f32 %f16, %l{i};
                    add.f32 %f16, %f16, %m{i};
                    mul.f32 %f16, %f16, 0f3F317218;
                    selp.f32 %f16, %f16, 0fFF800000, %p11;
                    add.u32 %r43, %r11, %r40;
                    mul.wide.u32 %rd12, %r43, 4;
                    add.u64 %rd12, %rd12, %rd7;
                    st.global.f32 [%rd12], %f16;
                WRITTEN{i}:
                """);
        }

        s.AppendLine("""
                ret;
            }
            """);
        sb.AppendLine(s.ToString());
    }
}
