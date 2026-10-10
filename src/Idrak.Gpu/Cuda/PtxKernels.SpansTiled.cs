// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;

namespace Idrak.Gpu.Cuda;

// Attention over one range of keys per query row (Backend.AttentionSpans), the fast kernels: one module per padded head
// size, built and loaded the first time a head size needs it (CudaBackend.Spans.cs), so a GPU loads only the sizes it
// runs. Two kinds:
//
// - attention_spans_f32_d{D} (float32, PTX 6.0 for sm_50 and newer): a block of 128 threads takes 64 query rows of one
//   head and walks the keys in tiles of 32. The products are tiled in registers (FlashAttention's outer-product form):
//   thread (tr, tc) = (tid / 8, tid % 8) holds the scores of rows 4tr .. 4tr + 3 and keys 4tc .. 4tc + 3 (16 sums: two
//   vector reads of shared memory per 16 multiply-adds), then the outputs of the same 4 rows and the columns tc + 8j.
//   Shared memory (dynamic): Qᵀ [D][64] (scaled by `scale` as it is staged), Kᵀ [D][32] (the same space then holds V
//   [32][D]), P [64][36] and the block's key range. The old attention_spans_f32 (one warp per 8 rows, a shared-memory
//   read per multiply-add: four times slower than the composed path where it was measured, plans/11-vision-language.md)
//   stays for head sizes this one does not take (not a multiple of 4) and devices whose shared memory per block cannot
//   hold this one's.
//
// - attention_spans_tc_d{D}_w{W} (bfloat16 tensor cores, PTX 7.0 for sm_80 and newer; when MixedPrecision asks for
//   them, as the other tensor-core attention): FlashAttention-2 as flash_tc_fwd (PtxKernels.FlashTensorCore.cs): a block
//   of W warps takes 16·W query rows (16 per warp, their Q fragments kept in registers), the keys in tiles of 64 staged in
//   shared memory as bfloat16, S = Q·Kᵀ and O += P·V on mma.m16n8k16 with float32 sums, online softmax in the log2
//   domain. Head sizes padded to a multiple of 16 (zeros past the real size). W = 4 or 8: measured per device and shape
//   (CudaBackend.Tuning.cs), 4 before measuring.
//
// Both: query row i of head h sees keys st[t·rows + i] ≤ c < en[t·rows + i] (clamped to [0, keyRows]; floats holding
// integers), t = h / headsPerTable, key/value head h / group. The block walks the keys from the smallest start of its
// rows (those that see any key) to their largest end; a key outside a row's range scores -∞ for it, and a row whose range
// is empty keeps a sum of 0: zeros, and a log-sum-exp of -∞. Head sizes up to 128 that are multiples of 4 (rows read as
// float4). Grid: x = ⌈rows / rows per block⌉, y = heads.
internal static partial class PtxKernels
{
    /// <summary>Query rows per block of the float32 span kernel.</summary>
    public const int SpanFloatRows = 64;

    /// <summary>Key positions per tile of the float32 span kernel.</summary>
    public const int SpanFloatKeys = 32;

    /// <summary>Key positions per tile of the tensor-core span kernel.</summary>
    public const int SpanTensorKeys = 64;

    // Row stride of the float32 kernel's P tile (floats): a multiple of 4 for its vector reads and writes.
    private const int SpanPStride = SpanFloatKeys + 4;

    /// <summary>Warps per block the tensor-core span kernel is built for (16 query rows each); the first is the default.</summary>
    public static readonly int[] SpanTensorWarps = [4, 8];

    /// <summary>The padded head size of the float32 span kernel for <paramref name="dim"/> (a multiple of 8), or 0 when it does not take it.</summary>
    public static int SpanFloatDim(int dim) => dim > 0 && dim % 4 == 0 && dim <= FlashMaxDim ? (dim + 7) / 8 * 8 : 0;

    /// <summary>The padded head size of the tensor-core span kernel for <paramref name="dim"/> (a multiple of 16), or 0 when it does not take it.</summary>
    public static int SpanTensorDim(int dim) => dim > 0 && dim % 4 == 0 && dim <= FlashMaxDim ? (dim + 15) / 16 * 16 : 0;

    /// <summary>The float32 span kernel's name for padded head size <paramref name="d"/>.</summary>
    public static string SpanFloatName(int d) => string.Create(CultureInfo.InvariantCulture, $"attention_spans_f32_d{d}");

    /// <summary>The tensor-core span kernel's name for padded head size <paramref name="d"/> and <paramref name="warps"/> warps.</summary>
    public static string SpanTensorName(int d, int warps) => string.Create(CultureInfo.InvariantCulture, $"attention_spans_tc_d{d}_w{warps}");

    /// <summary>Dynamic shared memory of the float32 span kernel (bytes): Qᵀ, Kᵀ (then V), P and the block's range.</summary>
    public static int SpanFloatShared(int d) => 4 * (SpanFloatRows * d + SpanFloatKeys * d + SpanFloatRows * SpanPStride + 8);

    /// <summary>The float32 span module for padded head size <paramref name="d"/> (PTX 6.0, sm_50, as the main module).</summary>
    public static (string Name, string[] Kernels, string Source) SpanFloatModule(int d) =>
        Module(string.Create(CultureInfo.InvariantCulture, $"attention spans f32 d{d}"), sb => SpanFloat(sb, d), version: "6.0", target: "sm_50");

    /// <summary>The tensor-core span module for padded head size <paramref name="d"/> (PTX 7.0, sm_80; every warp count).</summary>
    public static (string Name, string[] Kernels, string Source) SpanTensorModule(int d) =>
        Module(string.Create(CultureInfo.InvariantCulture, $"attention spans tc d{d}"), sb =>
        {
            foreach (int warps in SpanTensorWarps)
            {
                SpanTensor(sb, d, warps);
            }
        });

    // The parameters of both kernels (as attention_spans_f32's; the tensor-core kernel reads p_scale as scale · log2 e and
    // takes no soft-cap).
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

    private static void SpanFloat(StringBuilder sb, int d)
    {
        string name = SpanFloatName(d);
        int j = d / 8;                                 // output columns per thread: tc + 8·jj
        int qBytes = 4 * SpanFloatRows * d, kBytes = 4 * SpanFloatKeys * d, pBytes = 4 * SpanFloatRows * SpanPStride;
        int quads = d / 4;
        var s = new StringBuilder();
        s.AppendLine(CultureInfo.InvariantCulture, $$"""
            .extern .shared .align 16 .b8 {{name}}_smem[];
            .visible .entry {{name}}(
                {{SpanParameters}}, .param .f32 p_softcap
            )
            {
                .reg .pred %p<16>;
                .reg .b32 %r<64>;
                .reg .b64 %rd<24>;
                .reg .f32 %f<24>;
                .reg .f32 %s<16>;
                .reg .f32 %o<{{4 * j}}>;
                .reg .f32 %m<4>;
                .reg .f32 %l<4>;
                .reg .b32 %lo<4>;
                .reg .b32 %hi<4>;
                .reg .f32 %qv<4>;
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
                ld.param.f32 %softcap, [p_softcap];
                setp.gt.f32 %capped, %softcap, 0f00000000;
                mov.u32 %r6, %tid.x;
                and.b32 %r7, %r6, 7;
                shr.u32 %r8, %r6, 3;
                mov.u32 %r9, %ctaid.x;
                shl.b32 %r9, %r9, 6;
                mov.u32 %r10, %ctaid.y;
                mul.lo.u32 %r11, %r10, %r1;
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
                div.u32 %r13, %r10, %r5;
                mul.lo.u32 %r13, %r13, %r1;
                mov.u32 %r14, {{name}}_smem;
                add.u32 %r15, %r14, {{qBytes}};
                add.u32 %r16, %r15, {{kBytes}};
                add.u32 %r17, %r16, {{pBytes}};
            """);

        // Each thread's rows 4tr + i: their ranges, running maxima and sums.
        for (int i = 0; i < 4; i++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"""
                    shl.b32 %r19, %r8, 2;
                    add.u32 %r19, %r19, %r9;
                    add.u32 %r19, %r19, {i};
                {SpanRange("%r19", string.Create(CultureInfo.InvariantCulture, $"%lo{i}"), string.Create(CultureInfo.InvariantCulture, $"%hi{i}"), "%r1", "%r2", "%r13", "%rd4", "%rd5", "%r18", "%rd9", "%rd10", "%f2", "%f3", "%p1")}
                    mov.f32 %m{i}, 0fFF800000;
                    mov.f32 %l{i}, 0f00000000;
                """);
        }

        for (int i = 0; i < 4 * j; i++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    mov.f32 %o{i}, 0f00000000;");
        }

        // The block's keys (thread t reads row t % 64's range).
        s.AppendLine($"""
                and.b32 %r19, %r6, 63;
                add.u32 %r19, %r19, %r9;
            {SpanRange("%r19", "%r22", "%r23", "%r1", "%r2", "%r13", "%rd4", "%rd5", "%r18", "%rd9", "%rd10", "%f2", "%f3", "%p1")}
                and.b32 %r24, %r6, 31;
                shr.u32 %r25, %r6, 5;
            """);

        // Qᵀ[d][r] = scale · q[row0 + r, d], zeros past the rows and the head size: thread t takes row t % 64 of the quads
        // t / 64, +2, ... (a warp writes 32 consecutive floats of a Qᵀ row).
        for (int it = 0; it < d / 8; it++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $$"""
                    add.u32 %r26, %r6, {{it * 128}};
                    and.b32 %r27, %r26, 63;
                    shr.u32 %r28, %r26, 6;
                    shl.b32 %r28, %r28, 2;
                    add.u32 %r29, %r9, %r27;
                    setp.lt.u32 %p1, %r29, %r1;
                    setp.lt.and.u32 %p1, %r28, %r3, %p1;
                    cvt.u64.u32 %rd9, %r28;
                    mad.wide.u32 %rd9, %r29, %r3, %rd9;
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
                    mad.lo.u32 %r30, %r28, {{SpanFloatRows}}, %r27;
                    shl.b32 %r30, %r30, 2;
                    add.u32 %r30, %r30, %r14;
                    st.shared.f32 [%r30], %f4;
                    st.shared.f32 [%r30+{{4 * SpanFloatRows}}], %f5;
                    st.shared.f32 [%r30+{{8 * SpanFloatRows}}], %f6;
                    st.shared.f32 [%r30+{{12 * SpanFloatRows}}], %f7;
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
                    mad.wide.u32 %rd9, %r29, %r3, %rd9;
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
        for (int i = 0; i < 16; i++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    mov.f32 %s{i}, 0f00000000;");
        }

        // S[4tr + i, 4tc + jj] = Σ_d Qᵀ[d][4tr + i] · Kᵀ[d][4tc + jj], eight d per loop step.
        s.AppendLine(CultureInfo.InvariantCulture, $"""
                shl.b32 %r32, %r8, 4;
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
            s.AppendLine(CultureInfo.InvariantCulture, $"    ld.shared.v4.f32 {{%qv0, %qv1, %qv2, %qv3}}, [%r32+{u * 4 * SpanFloatRows}];");
            s.AppendLine(CultureInfo.InvariantCulture, $"    ld.shared.v4.f32 {{%kv0, %kv1, %kv2, %kv3}}, [%r33+{u * 4 * SpanFloatKeys}];");
            for (int i = 0; i < 4; i++)
            {
                for (int c = 0; c < 4; c++)
                {
                    s.AppendLine(CultureInfo.InvariantCulture, $"    fma.rn.f32 %s{4 * i + c}, %qv{i}, %kv{c}, %s{4 * i + c};");
                }
            }
        }

        s.AppendLine(CultureInfo.InvariantCulture, $"""
                add.u32 %r32, %r32, {8 * 4 * SpanFloatRows};
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
                    mad.wide.u32 %rd9, %r29, %r3, %rd9;
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
        for (int i = 0; i < 4; i++)
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

            // P[4tr + i][4tc .. 4tc + 3]
            s.AppendLine(CultureInfo.InvariantCulture, $$"""
                    shl.b32 %r37, %r8, 2;
                    add.u32 %r37, %r37, {{i}};
                    mul.lo.u32 %r37, %r37, {{SpanPStride}};
                    add.u32 %r37, %r37, %r35;
                    sub.u32 %r37, %r37, %r31;
                    shl.b32 %r37, %r37, 2;
                    add.u32 %r37, %r37, %r16;
                    st.shared.v4.f32 [%r37], {%s{{4 * i}}, %s{{4 * i + 1}}, %s{{4 * i + 2}}, %s{{4 * i + 3}}};
                """);
        }

        // O[4tr + i, tc + 8c] += Σ_key P[4tr + i, key] · V[key, tc + 8c], four keys per loop step (P read as float4).
        s.AppendLine(CultureInfo.InvariantCulture, $"""
                bar.sync 0;
                shl.b32 %r38, %r8, 2;
                mul.lo.u32 %r38, %r38, {4 * SpanPStride};
                add.u32 %r38, %r38, %r16;
                shl.b32 %r39, %r7, 2;
                add.u32 %r39, %r39, %r15;
                mov.u32 %r34, 0;
            PV:
                setp.ge.u32 %p5, %r34, {SpanFloatKeys / 4};
                @%p5 bra PV_END;
            """);
        for (int i = 0; i < 4; i++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    ld.shared.v4.f32 {{%s{4 * i}, %s{4 * i + 1}, %s{4 * i + 2}, %s{4 * i + 3}}}, [%r38+{i * 4 * SpanPStride}];");
        }

        for (int u = 0; u < 4; u++)
        {
            for (int c = 0; c < j; c++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    ld.shared.f32 %f12, [%r39+{u * 4 * d + c * 32}];");
                for (int i = 0; i < 4; i++)
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
        for (int i = 0; i < 4; i++)
        {
            foreach (int offset in new[] { 1, 2, 4 })
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    shfl.sync.bfly.b32 %f13, %l{i}, {offset}, 31, 0xffffffff;");
                s.AppendLine(CultureInfo.InvariantCulture, $"    add.f32 %l{i}, %l{i}, %f13;");
            }

            s.AppendLine(CultureInfo.InvariantCulture, $"""
                    shl.b32 %r40, %r8, 2;
                    add.u32 %r40, %r40, %r9;
                    add.u32 %r40, %r40, {i};
                    setp.ge.u32 %p9, %r40, %r1;
                    @%p9 bra WRITTEN{i};
                    setp.gt.f32 %p11, %l{i}, 0f00000000;
                    rcp.rn.f32 %f14, %l{i};
                    selp.f32 %f14, %f14, 0f00000000, %p11;
                    mad.lo.u32 %r41, %r40, %r3, %r7;
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

    // ------------------------------------------------------------------ tensor cores

    // Stages `rows` rows (absolute first + 0 … rows - 1; zeros at or past `limit` and past the head size in %r22) of a
    // float32 [*, dim] tensor at `pointer` into shared memory at `shared` as bfloat16 ([rows][d + 8]); `threads` threads,
    // a float4 each per step, along the rows. Steps in a loop of four at a time (`label` names it) and the rest after it,
    // so a large tile does not keep every step's loads in registers at once. Uses %r11-%r19, %r53, %rd1-%rd2, %f0-%f3,
    // %p1, %p20.
    private static string SpanTensorStage(int rows, int d, int threads, string shared, string first, string limit, string pointer, string label)
    {
        const int Unroll = 4;
        int quads = d / 4, stride = FlashStride(d), elements = rows * quads, iterations = (elements + threads - 1) / threads;
        int looped = iterations > Unroll ? iterations / Unroll * Unroll : 0;
        var s = new StringBuilder();

        // One step: element e = `index` (+ `offset` threads), row e / quads, float4 e % quads.
        void Step(string index, int offset, bool guard)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $$"""
                    add.u32 %r11, {{index}}, {{offset}};
                    div.u32 %r12, %r11, {{quads}};
                    rem.u32 %r13, %r11, {{quads}};
                    add.u32 %r14, %r12, {{first}};
                    shl.b32 %r15, %r13, 2;
                    setp.lt.u32 %p1, %r14, {{limit}};
                    setp.lt.and.u32 %p1, %r15, %r22, %p1;
                    {{(guard ? string.Create(CultureInfo.InvariantCulture, $"setp.lt.and.u32 %p1, %r11, {elements}, %p1;") : "")}}
                    cvt.u64.u32 %rd2, %r15;
                    mad.wide.u32 %rd1, %r14, %r22, %rd2;
                    shl.b64 %rd1, %rd1, 2;
                    add.u64 %rd1, %rd1, {{pointer}};
                    mov.f32 %f0, 0f00000000;
                    mov.f32 %f1, 0f00000000;
                    mov.f32 %f2, 0f00000000;
                    mov.f32 %f3, 0f00000000;
                    @%p1 ld.global.v4.f32 {%f0, %f1, %f2, %f3}, [%rd1];
                    cvt.rn.bf16x2.f32 %r16, %f1, %f0;
                    cvt.rn.bf16x2.f32 %r17, %f3, %f2;
                    mad.lo.u32 %r18, %r12, {{stride}}, {{shared}};
                    shl.b32 %r19, %r13, 3;
                    add.u32 %r18, %r18, %r19;
                """);
            s.AppendLine(guard
                ? string.Create(CultureInfo.InvariantCulture, $"setp.lt.u32 %p1, %r11, {elements};\n@%p1 st.shared.v2.b32 [%r18], {{%r16, %r17}};")
                : "st.shared.v2.b32 [%r18], {%r16, %r17};");
        }

        if (looped > 0)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"""
                    mov.u32 %r53, %r1;
                {label}:
                    setp.ge.u32 %p20, %r53, {looped * threads};
                    @%p20 bra {label}_END;
                """);
            for (int u = 0; u < Unroll; u++)
            {
                Step("%r53", u * threads, guard: false);
            }

            s.AppendLine(CultureInfo.InvariantCulture, $"""
                    add.u32 %r53, %r53, {Unroll * threads};
                    bra {label};
                {label}_END:
                """);
        }

        for (int it = looped; it < iterations; it++)
        {
            Step("%r1", it * threads, guard: (it + 1) * threads > elements);
        }

        return s.ToString();
    }

    private static void SpanTensor(StringBuilder sb, int d, int warps)
    {
        int stride = FlashStride(d), dTiles = d / 8, kSteps = d / 16, threads = 32 * warps, rowsPerBlock = 16 * warps;
        int tileBytes = SpanTensorKeys * stride;
        string name = SpanTensorName(d, warps);
        var s = new StringBuilder();
        s.AppendLine(CultureInfo.InvariantCulture, $$"""
            .visible .entry {{name}}(
                {{SpanParameters}}
            )
            {
            {{FlashRegisters(d, 4 * dTiles)}}
                .shared .align 16 .b8 {{name}}_kv[{{Math.Max(2 * tileBytes, rowsPerBlock * stride)}}];
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
                mov.u32 %r24, %ctaid.y;
                mov.u32 %r25, %ctaid.x;
                mul.lo.u32 %r25, %r25, {{rowsPerBlock}};
                mul.lo.u32 %r44, %r24, %r20;
                mul.wide.u32 %rd40, %r44, %r22;
                shl.b64 %rd40, %rd40, 2;
                add.u64 %rd10, %rd10, %rd40;
                add.u64 %rd14, %rd14, %rd40;
                div.u32 %r44, %r24, %r23;
                mul.lo.u32 %r44, %r44, %r21;
                mul.wide.u32 %rd41, %r44, %r22;
                shl.b64 %rd41, %rd41, 2;
                add.u64 %rd11, %rd11, %rd41;
                add.u64 %rd12, %rd12, %rd41;
                div.u32 %r28, %r24, %r28;
                mul.lo.u32 %r28, %r28, %r20;
                mov.u32 %r26, {{name}}_kv;
                add.u32 %r27, %r26, {{tileBytes}};
                mov.u32 %r48, {{name}}_range;
            """);

        // This thread's rows (g and g + 8 of its warp's 16): their ranges; then the block's keys.
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

        // Q tile → shared → fragments (the range reduction's barrier also orders the staging).
        s.AppendLine(SpanTensorStage(rowsPerBlock, d, threads, "%r26", "%r25", "%r20", "%rd10", "QS"));
        s.AppendLine(SpanBlockRange("%r46", "%r47", "%r21", "%r2", "%r3", warps, "%r48", "%r36", "%r37", "%r44", "%r45", "%p3"));
        s.AppendLine(CultureInfo.InvariantCulture, $"""
                shl.b32 %r29, %r3, 4;
                mul.lo.u32 %r29, %r29, {stride};
                add.u32 %r29, %r29, %r26;
                add.u32 %r29, %r29, %r6;
            """);
        for (int ks = 0; ks < kSteps; ks++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%qa{4 * ks}, %qa{4 * ks + 1}, %qa{4 * ks + 2}, %qa{4 * ks + 3}}}, [%r29+{ks * 32}];");
        }

        s.AppendLine($"""
                bar.sync 0;
                mov.f32 %f42, 0fFF800000;
                mov.f32 %f43, 0fFF800000;
                mov.f32 %f44, 0f00000000;
                mov.f32 %f45, 0f00000000;
                add.u32 %r40, %r27, %r6;
                add.u32 %r41, %r26, %r7;
                shl.b32 %r42, %r5, 1;
            {Zeros("%acc", 4 * dTiles)}
                mov.u32 %r43, %r36;
            TILE:
                setp.ge.u32 %p3, %r43, %r37;
                @%p3 bra TILE_END;
            """);
        s.AppendLine(SpanTensorStage(SpanTensorKeys, d, threads, "%r26", "%r43", "%r37", "%rd11", "KS"));
        s.AppendLine(SpanTensorStage(SpanTensorKeys, d, threads, "%r27", "%r43", "%r37", "%rd12", "VS"));
        s.AppendLine("bar.sync 0;");
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

        // Scale, mask by each row's range (key c = tile + 8 nt + 2t + j; row g: [%r32, %r33), g + 8: [%r34, %r35)), maxima.
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

        // The row sums stay per thread (the quad's four partial sums are added once, at the end).
        s.AppendLine("""
                fma.rn.f32 %f44, %f44, %f51, %f53;
                fma.rn.f32 %f45, %f45, %f52, %f54;
            """);
        for (int nt = 0; nt < dTiles; nt++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"""
                    mul.f32 %acc{4 * nt}, %acc{4 * nt}, %f51;
                    mul.f32 %acc{4 * nt + 1}, %acc{4 * nt + 1}, %f51;
                    mul.f32 %acc{4 * nt + 2}, %acc{4 * nt + 2}, %f52;
                    mul.f32 %acc{4 * nt + 3}, %acc{4 * nt + 3}, %f52;
                """);
        }

        for (int jj = 0; jj < 4; jj++)
        {
            s.AppendLine(PackA("%s", jj));
            for (int dn2 = 0; dn2 < d / 16; dn2++)
            {
                s.AppendLine(LdMatrix("%tb", "%r40", jj * 16 * stride + dn2 * 32, transpose: true));
                s.AppendLine(Mma("%acc", 2 * dn2, "%pa", "%tb0", "%tb1"));
                s.AppendLine(Mma("%acc", 2 * dn2 + 1, "%pa", "%tb2", "%tb3"));
            }
        }

        s.AppendLine(CultureInfo.InvariantCulture, $"""
                bar.sync 0;
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
                mad.lo.u32 %r51, %r30, %r22, %r42;
                mul.wide.u32 %rd22, %r51, 4;
                add.u64 %rd22, %rd22, %rd14;
                mad.lo.u32 %r51, %r31, %r22, %r42;
                mul.wide.u32 %rd26, %r51, 4;
                add.u64 %rd26, %rd26, %rd14;
            """);
        for (int nt = 0; nt < dTiles; nt++)
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
