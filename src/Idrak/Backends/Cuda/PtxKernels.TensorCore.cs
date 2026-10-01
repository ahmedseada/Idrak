// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;

namespace Idrak.Backends.Cuda;

/// <summary>
/// Tensor-core matrix multiplies (sm_80 and newer: Ampere, Ada, Hopper, Blackwell), in a module of their own so older
/// GPUs still load the main one. Inputs and outputs are float32 like every other kernel: tiles are rounded to bfloat16
/// as they are copied to shared memory, multiplied with <c>mma.sync.m16n8k16</c> and summed in float32 (what PyTorch's
/// bfloat16 autocast does, without the separate cast kernels and copies).
/// </summary>
internal static partial class PtxKernels
{
    /// <summary>Output tile edge of the tensor-core GEMM (128 × 128 per block of 256 threads, 32 k per stage).</summary>
    public const int TensorTile = 128;

    /// <summary>Threads per tensor-core GEMM block.</summary>
    public const int TensorThreads = 256;

    private const int TensorK = 32;

    // Shared-memory row strides in bytes (bfloat16 rows padded by 8 elements, so ldmatrix's eight 16-byte row reads
    // fall in distinct banks): tiles 32 wide (k contiguous) and 128 wide (m or n contiguous).
    private const int NarrowStride = (TensorK + 8) * 2, WideStride = (TensorTile + 8) * 2;
    private const int StageBytes = TensorTile * NarrowStride;           // the larger of 128 × 80 and 32 × 272

    /// <summary>
    /// The tensor-core modules (PTX 7.0, sm_80), loaded separately so a GPU driver that rejects one still gets the others:
    /// products (gemm_tc_{a}{b}_f32 with a / b = n (as stored) or t (transposed), plus the GELU and GELU-gradient
    /// epilogue variants), flash attention for head sizes 64 and 128, and the attention backward's row sums.
    /// </summary>
    public static IReadOnlyList<(string Name, string[] Kernels, string Source)> TensorCoreModules => LazyTensorCoreModules.Value;

    private static readonly Lazy<IReadOnlyList<(string Name, string[] Kernels, string Source)>> LazyTensorCoreModules = new(() =>
    [
        Module("products", sb =>
        {
            foreach (bool ta in new[] { false, true })
            {
                foreach (bool tb in new[] { false, true })
                {
                    TensorCoreGemm(sb, ta, tb, 0);
                }
            }

            TensorCoreGemm(sb, false, false, 1);
            TensorCoreGemm(sb, false, true, 2);
            TensorCoreGemm(sb, false, false, 0, packed: 1);
            TensorCoreGemm(sb, false, false, 0, packed: 2);
            TensorCoreGemm(sb, false, false, 0, packed: 3);
            TensorCoreGemm(sb, false, false, 0, packed: 1, multi: true);
            TensorCoreGemm(sb, false, false, 0, packed: 2, multi: true);
            TensorCoreGemm(sb, false, false, 0, packed: 3, multi: true);
            foreach (int packed in new[] { 1, 2, 3 })
            {
                TensorCoreGemm(sb, false, false, 0, packed: packed, tileM: 64);
                TensorCoreGemm(sb, false, false, 0, packed: packed, multi: true, tileM: 64);
            }

            // LoRA: the adapter's low-rank term as one more k step of the product (…_lr_f32).
            TensorCoreGemm(sb, false, false, 0, lowRank: true);
            TensorCoreGemm(sb, false, true, 0, lowRank: true);
            TensorCoreGemm(sb, false, true, 0, lowRank: true, bf16B: true);     // dx = g · Wᵀ (+ dt · Aᵀ) over bfloat16 weights
            foreach (int packed in new[] { 2, 3 })
            {
                TensorCoreGemm(sb, false, false, 0, packed: packed, lowRank: true);
                TensorCoreGemm(sb, false, false, 0, packed: packed, multi: true, lowRank: true);
            }
        }),
        Module("attention d64", sb => { FlashForward(sb, 64); FlashBackwardQ(sb, 64); FlashBackwardKv(sb, 64); }),
        Module("attention d128", sb => { FlashForward(sb, 128); FlashBackwardQ(sb, 128); FlashBackwardKv(sb, 128); }),
        Module("attention row sums", FlashDelta),
        Module("int8 products", sb => BuildEightBit(sb, fp8: false)),
        Module("fp8 products", sb => BuildEightBit(sb, fp8: true), version: "8.4", target: "sm_89"),
    ]);

    private static (string Name, string[] Kernels, string Source) Module(string name, Action<StringBuilder> build, string version = "7.0", string target = "sm_80")
    {
        var sb = new StringBuilder();
        sb.AppendLine($".version {version}");
        sb.AppendLine($".target {target}");
        sb.AppendLine(".address_size 64");
        sb.AppendLine();
        build(sb);
        string source = sb.ToString();
        string[] kernels = [.. System.Text.RegularExpressions.Regex.Matches(source, @"\.entry\s+(\w+)").Select(m => m.Groups[1].Value)];
        return (name, kernels, source);
    }

    /// <summary>Every tensor-core kernel.</summary>
    public static string[] TensorCoreNames => [.. TensorCoreModules.SelectMany(m => m.Kernels)];

    /// <summary>All tensor-core modules' PTX (for dumps and signature checks).</summary>
    public static string TensorCoreSource => string.Concat(TensorCoreModules.Select(m => m.Source));

    private static readonly Lazy<IReadOnlyDictionary<string, int>> LazyTensorCoreParameterCounts = new(() =>
        System.Text.RegularExpressions.Regex.Matches(TensorCoreSource, @"\.entry\s+(\w+)\s*\(([^)]*)\)")
            .ToDictionary(m => m.Groups[1].Value, m => System.Text.RegularExpressions.Regex.Count(m.Groups[2].Value, @"\.param\b")));

    /// <summary>Parameters each tensor-core kernel declares.</summary>
    public static IReadOnlyDictionary<string, int> TensorCoreParameterCounts => LazyTensorCoreParameterCounts.Value;

    /// <summary>The product kernel for the layouts and epilogue (null when there is no such variant).</summary>
    public static string? GemmKernel(bool transA, bool transB, GemmEpilogue epilogue) => epilogue switch
    {
        GemmEpilogue.None => $"gemm_tc_{(transA ? 't' : 'n')}{(transB ? 't' : 'n')}_f32",
        GemmEpilogue.Gelu when !transA && !transB => "gemm_tc_nn_gelu_f32",
        GemmEpilogue.GeluGradient when !transA && transB => "gemm_tc_nt_gelugrad_f32",
        _ => null,
    };

    // One operand's tile as it is read from global memory: `outer` rows of `width` contiguous floats (the stored layout),
    // copied to shared memory in the same orientation. A as stored [m, k] is 128 rows of 32 k; transposed ([k, m]) 32
    // rows of 128 m. B as stored [k, n] is 32 rows of 128 n; transposed ([n, k]) 128 rows of 32 k.
    private sealed record TileLoad(string Name, bool KIsOuter, int Width, string OuterLimit, string InnerLimit, string Tile, int Rows = TensorTile)
    {
        public int PairsPerRow => Width / 2;
        public int RowStep => 2 * TensorThreads / Width;                  // rows between one thread's successive pairs
        public int Pairs => Rows * TensorK / 2 / TensorThreads;           // 8 pairs per thread (4 for a 64-row A tile)
        public int Stride => Width == TensorK ? NarrowStride : WideStride;
    }

    // GELU (tanh approximation, as gelu_f32): %t0 = x², %t5 = tanh(√(2/π) (x + 0.044715 x³)); uses %t0-%t5.
    private static string GeluTerms(string x) => $"""
            mul.f32 %t0, {x}, {x};
            mul.f32 %t1, %t0, {x};
            fma.rn.f32 %t2, %t1, {F(0.044715f)}, {x};
            mul.f32 %t2, %t2, {F(0.7978845608f)};
            mul.f32 %t3, %t2, {F(2.8853900817779268f)};
            ex2.approx.ftz.f32 %t3, %t3;
            add.f32 %t3, %t3, 0f3F800000;
            rcp.rn.f32 %t4, %t3;
            fma.rn.f32 %t5, %t4, 0fC0000000, 0f3F800000;
        """;

    // mode (compile time): 0 plain, 1 GELU (aux = pre-activations when given), 2 GELU gradient (· gelu'(aux)).
    // packed (nn, mode 0 only): 1 = int8 weights (4 per word, per-column scales in p_aux, applied in the epilogue),
    // 2 = 4-bit weights (8 per word, scales per 32 rows and column in p_aux, applied as the tile is unpacked), 3 = bfloat16
    // weights (2 per word); ldb is then the row length in words. The weights are unpacked into the bfloat16 tile.
    //
    // multi (packed, …_multi_f32): up to three products of the same input (queries/keys/values, gate/up), each with its
    // own weights, scales and output: p_b/p_c/p_aux/p_n for the first, then p_b1… and p_b2… (p_n2 = 0 for two). Column
    // tiles of product 0 come first, then 1, then 2 (widths multiples of 128); ldb and ldc follow each product's width.
    //
    // tileM 64 (packed, …_m64_f32): 64-row tiles for prompts whose last 128-row tile would be mostly empty (180 rows: 192
    // computed instead of 256); the 8 warps take 32 × 32 each instead of 64 × 32.
    //
    // lowRank (A as stored, mode 0, 128-row tiles, not int8, …_lr_f32): c = beta·c + a·b + u·v, a LoRA adapter's term
    // added as one more k step of at most 32: u [m, r] (p_u, rows of r floats), v [r, n] (b as stored; rows of n floats)
    // or [n, r] (b transposed), r = p_r ≤ 32; multi takes p_u1 / p_v1 and p_u2 / p_v2 for its other products. p_u = 0
    // skips the term. With split k only block z = 0 adds it.
    //
    // bf16B (B transposed, …_nt_bf16w_lr_f32): B is a bfloat16 weight W [n][k] stored as BFloat16Weight packs it (two
    // values per 32-bit word along k; ldb = words per row): dx = g · Wᵀ reads the weight as stored, with no float copy. Each
    // thread's pair of k is one word, copied to shared memory as it is (the tile holds bfloat16 pairs anyway).
    private static void TensorCoreGemm(StringBuilder sb, bool ta, bool tb, int mode, int packed = 0, bool multi = false, int tileM = TensorTile,
        bool lowRank = false, bool bf16B = false)
    {
        if (lowRank && (ta || mode != 0 || packed == 1 || tileM != TensorTile))
        {
            throw new ArgumentException("The low-rank stage needs A as stored, no epilogue, no int8 column scales and 128-row tiles.");
        }

        if (bf16B && (!tb || ta || mode != 0 || packed != 0 || multi))
        {
            throw new ArgumentException("bfloat16 B words need B transposed, A as stored, no epilogue and no other packing.");
        }

        string name = $"gemm_tc_{(ta ? 't' : 'n')}{(tb ? 't' : 'n')}{mode switch { 1 => "_gelu", 2 => "_gelugrad", _ => "" }}"
                      + $"{packed switch { 1 => "_int8w", 2 => "_int4w", 3 => "_bf16w", _ => "" }}{(bf16B ? "_bf16w" : "")}{(multi ? "_multi" : "")}{(tileM == 64 ? "_m64" : "")}{(lowRank ? "_lr" : "")}_f32";
        int mts = tileM / 32, warpRowShift = (int)Math.Log2(tileM / 2);          // m16 slices per warp; the warp's first row
        int cpw = packed switch { 1 => 4, 2 => 8, _ => 2 }, tileWords = TensorTile / cpw, wordRows = TensorThreads / tileWords;
        int wordsPerThread = TensorK * tileWords / TensorThreads;
        // Registers: %rd1..3 = a, b, c; %r1..3 = m, n, k; %r9 / %r10 = the tile's first row / column.
        var a = new TileLoad("a", KIsOuter: ta, Width: ta ? TensorTile : TensorK, OuterLimit: ta ? "%r3" : "%r1", InnerLimit: ta ? "%r1" : "%r3", Tile: "%r9",
            Rows: ta ? TensorTile : tileM);
        var b = new TileLoad("b", KIsOuter: !tb, Width: tb ? TensorK : TensorTile, OuterLimit: tb ? "%r2" : "%r3", InnerLimit: tb ? "%r3" : "%r2", Tile: "%r10");
        var s = new StringBuilder();
        s.AppendLine($$"""
            .visible .entry {{name}}(
                .param .u64 p_a, .param .u64 p_b, .param .u64 p_c,
                .param .u32 p_m, .param .u32 p_n, .param .u32 p_k, .param .f32 p_beta,
                .param .u64 p_sa, .param .u64 p_sb, .param .u64 p_sc, .param .u64 p_bias,
                .param .u32 p_lda, .param .u32 p_ldb, .param .u32 p_ldc, .param .u64 p_aux{{(multi ? """
                ,
                                .param .u64 p_b1, .param .u64 p_c1, .param .u64 p_aux1, .param .u32 p_n1,
                                .param .u64 p_b2, .param .u64 p_c2, .param .u64 p_aux2, .param .u32 p_n2
                """ : "")}}{{(lowRank ? """
                ,
                                .param .u64 p_u, .param .u64 p_v, .param .u32 p_r
                """ : "")}}{{(lowRank && multi ? """
                ,
                                .param .u64 p_u1, .param .u64 p_v1, .param .u64 p_u2, .param .u64 p_v2
                """ : "")}}
            )
            {
                .reg .pred %p<16>;
                .reg .u64 %rdaux;
                .reg .pred %pm0;
                .reg .pred %pbeta, %peven, %qnot;
                .reg .pred %q<12>;
                .reg .f32 %e<8>;
                .reg .f32 %h<8>;
                .reg .f32 %t<8>;
                .reg .f32 %bias<8>;
                .reg .pred %pbias, %paux, %psplit, %pz0;
                .reg .b32 %gw<8>;
                .reg .f32 %gs<16>;
                .reg .f32 %gx<8>;
                .reg .f32 %c<64>;
                .reg .b32 %fa<16>;
                .reg .b32 %fb<8>;
                .reg .f32 %ga<16>;
                .reg .f32 %gb<16>;
                .reg .f32 %f<4>;
                .reg .f32 %beta;
                .reg .b32 %r<64>;
                .reg .b64 %rd<32>;
                .reg .b32 %lr<8>;
                .reg .b64 %lrd<8>;
                .reg .pred %plr;
                .shared .align 16 .b8 {{name}}_as[{{2 * StageBytes}}];
                .shared .align 16 .b8 {{name}}_bs[{{2 * StageBytes}}];
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
                ld.param.u64 %rd4, [p_sa];
                ld.param.u64 %rd5, [p_sb];
                ld.param.u64 %rd6, [p_sc];
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
                mov.u32 %r4, %tid.x;
                and.b32 %r5, %r4, 31;
                shr.u32 %r6, %r4, 5;
                and.b32 %r7, %r6, 1;
                shr.u32 %r8, %r6, 1;
                // Tile order grouped by 8 row tiles (blocks that run together share B's column tiles in L2): linear
                // block b, group of 8*gx blocks, row tile first + b % size, column tile (b % (8*gx)) / size.
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
                shl.b32 %r9, %r9, {{(int)Math.Log2(tileM)}};
                shl.b32 %r10, %r10, 7;
                mov.u32 %r11, {{name}}_as;
                mov.u32 %r12, {{name}}_bs;
                ld.param.u32 %r58, [p_lda];
                ld.param.u32 %r59, [p_ldb];
                ld.param.u32 %r60, [p_ldc];
                ld.param.u64 %rdaux, [p_aux];
            """);
        if (lowRank)
        {
            s.AppendLine("""
                    ld.param.u64 %lrd2, [p_u];
                    ld.param.u64 %lrd3, [p_v];
                """);
        }

        if (multi)
        {
            // This block's product: its column tile %r10 past the widths of the products before it.
            s.AppendLine($"""
                    setp.ge.u32 %pm0, %r10, %r2;
                    @!%pm0 bra PRODUCT_SET;
                    sub.u32 %r10, %r10, %r2;
                    ld.param.u64 %rd2, [p_b1];
                    ld.param.u64 %rd3, [p_c1];
                    ld.param.u64 %rdaux, [p_aux1];
                    ld.param.u32 %r2, [p_n1];
                    {(lowRank ? "ld.param.u64 %lrd2, [p_u1];" : "")}
                    {(lowRank ? "ld.param.u64 %lrd3, [p_v1];" : "")}
                    setp.ge.u32 %pm0, %r10, %r2;
                    @!%pm0 bra PRODUCT_POINTERS;
                    sub.u32 %r10, %r10, %r2;
                    ld.param.u64 %rd2, [p_b2];
                    ld.param.u64 %rd3, [p_c2];
                    ld.param.u64 %rdaux, [p_aux2];
                    ld.param.u32 %r2, [p_n2];
                    {(lowRank ? "ld.param.u64 %lrd2, [p_u2];" : "")}
                    {(lowRank ? "ld.param.u64 %lrd3, [p_v2];" : "")}
                PRODUCT_POINTERS:
                    cvta.to.global.u64 %rd2, %rd2;
                    cvta.to.global.u64 %rd3, %rd3;
                    add.u32 %r59, %r2, {cpw - 1};
                    shr.u32 %r59, %r59, {(int)Math.Log2(cpw)};
                    mov.u32 %r60, %r2;
                PRODUCT_SET:
                """);
        }
        if (mode == 0)
        {
            // Split k (gridDim.z > 1 on packed weights, or with p_sc = 0 on plain ones, whose z otherwise indexes a
            // batch): block z sums k [%r62, %r3) of chunks rounded to 32 (the int4 scale groups), added into c
            // atomically by the epilogue.
            s.AppendLine($"""
                    mov.u32 %r49, %nctaid.z;
                    setp.gt.u32 %psplit, %r49, 1;
                    {(packed != 0 ? "" : "setp.eq.and.u64 %psplit, %rd6, 0, %psplit;")}
                    mov.u32 %r62, 0;
                    @!%psplit bra KSPLIT_DONE;
                    add.u32 %r61, %r3, %r49;
                    sub.u32 %r61, %r61, 1;
                    div.u32 %r61, %r61, %r49;
                    add.u32 %r61, %r61, 31;
                    and.b32 %r61, %r61, 0xFFFFFFE0;
                    mul.lo.u32 %r62, %r40, %r61;
                    add.u32 %r63, %r62, %r61;
                    min.u32 %r3, %r63, %r3;
                KSPLIT_DONE:
                """);
        }


        // Per-thread load coordinates (row within the tile, first of its two columns) and shared-memory store address.
        // A: %r13 row, %r14 column, %r15 store address; B: %r16, %r17, %r18. Leading dimensions (bytes): %rd9 / %rd10.
        void Coordinates(TileLoad t, string row, string column, string store, string smem, string leading, string ld)
        {
            s.AppendLine($"""
                    shr.u32 {row}, %r4, {(int)Math.Log2(t.PairsPerRow)};
                    and.b32 {column}, %r4, {t.PairsPerRow - 1};
                    shl.b32 {column}, {column}, 1;
                    mul.lo.u32 {store}, {row}, {t.Stride};
                    shl.b32 %r19, {column}, 1;
                    add.u32 {store}, {store}, %r19;
                    add.u32 {store}, {store}, {smem};
                    mul.wide.u32 {leading}, {ld}, 4;
                """);
        }

        Coordinates(a, "%r13", "%r14", "%r15", "%r11", "%rd9", "%r58");
        if (packed == 0)
        {
            Coordinates(b, "%r16", "%r17", "%r18", "%r12", "%rd10", "%r59");
        }
        else
        {
            // Packed B: %r16 = row in the tile, %r17 = word within the tile's row, %r18 = its shared-memory address.
            s.AppendLine($"""
                    shr.u32 %r16, %r4, {(int)Math.Log2(tileWords)};
                    and.b32 %r17, %r4, {tileWords - 1};
                    mul.lo.u32 %r18, %r16, {WideStride};
                    shl.b32 %r19, %r17, {(int)Math.Log2(cpw * 2)};
                    add.u32 %r18, %r18, %r19;
                    add.u32 %r18, %r18, %r12;
                    mul.wide.u32 %rd10, %r59, 4;
                    ld.param.u64 %rd26, [p_aux];
                    cvta.to.global.u64 %rd26, %rd26;
                """);
        }

        // ldmatrix lane addresses (without the stage offset), %r20 for A and %r21 for B; see the fragment layouts of
        // mma.m16n8k16 (row-major A 16 × 16, column-major B 16 × 8).
        if (!ta)
        {
            // A stored [m][k]: lanes 0-15 rows 0-15 at k 0, lanes 16-31 rows 0-15 at k 8 (non-transposed ldmatrix).
            s.AppendLine($"""
                    and.b32 %r22, %r5, 15;
                    shl.b32 %r23, %r7, {warpRowShift};
                    add.u32 %r22, %r22, %r23;
                    mul.lo.u32 %r20, %r22, {NarrowStride};
                    shr.u32 %r23, %r5, 4;
                    shl.b32 %r23, %r23, 4;
                    add.u32 %r20, %r20, %r23;
                    add.u32 %r20, %r20, %r11;
                """);
        }
        else
        {
            // A stored [k][m]: lane l reads k row (l & 7) + 8 (l >> 4) at m offset 8 ((l >> 3) & 1) (transposing ldmatrix).
            s.AppendLine($"""
                    and.b32 %r22, %r5, 7;
                    shr.u32 %r23, %r5, 4;
                    shl.b32 %r23, %r23, 3;
                    add.u32 %r22, %r22, %r23;
                    mul.lo.u32 %r20, %r22, {WideStride};
                    shr.u32 %r23, %r5, 3;
                    and.b32 %r23, %r23, 1;
                    shl.b32 %r23, %r23, 3;
                    shl.b32 %r24, %r7, 6;
                    add.u32 %r23, %r23, %r24;
                    shl.b32 %r23, %r23, 1;
                    add.u32 %r20, %r20, %r23;
                    add.u32 %r20, %r20, %r11;
                """);
        }

        if (tb)
        {
            // B stored [n][k]: lane l reads n row (l & 7) + 8 (l >> 4) at k offset 8 ((l >> 3) & 1) (non-transposed).
            s.AppendLine($"""
                    and.b32 %r22, %r5, 7;
                    shr.u32 %r23, %r5, 4;
                    shl.b32 %r23, %r23, 3;
                    add.u32 %r22, %r22, %r23;
                    shl.b32 %r24, %r8, 5;
                    add.u32 %r22, %r22, %r24;
                    mul.lo.u32 %r21, %r22, {NarrowStride};
                    shr.u32 %r23, %r5, 3;
                    and.b32 %r23, %r23, 1;
                    shl.b32 %r23, %r23, 4;
                    add.u32 %r21, %r21, %r23;
                    add.u32 %r21, %r21, %r12;
                """);
        }
        else
        {
            // B stored [k][n]: lane l reads k row (l & 15) at n offset 8 (l >> 4) (transposing ldmatrix).
            s.AppendLine($"""
                    and.b32 %r22, %r5, 15;
                    mul.lo.u32 %r21, %r22, {WideStride};
                    shr.u32 %r23, %r5, 4;
                    shl.b32 %r23, %r23, 3;
                    shl.b32 %r24, %r8, 5;
                    add.u32 %r23, %r23, %r24;
                    shl.b32 %r23, %r23, 1;
                    add.u32 %r21, %r21, %r23;
                    add.u32 %r21, %r21, %r12;
                """);
        }

        for (int i = 0; i < 64; i++)
        {
            s.AppendLine($"    mov.f32 %c{i}, 0f00000000;");
        }

        // Global → registers for the k tile starting at %r30: 8 pairs of adjacent floats per operand, zero outside the matrix.
        void Load(TileLoad t, string baseAddress, string leading, string row, string column, string staging)
        {
            bool words = bf16B && t.Name == "b";                              // one bfloat16 pair per 32-bit word
            // outer / inner index of the thread's first pair.
            string outerBase = t.KIsOuter ? "%r30" : t.Tile, innerBase = t.KIsOuter ? t.Tile : "%r30";
            s.AppendLine($"""
                    add.u32 %r31, {outerBase}, {row};
                    add.u32 %r32, {innerBase}, {column};
                    setp.lt.u32 %p1, %r32, {t.InnerLimit};
                    add.u32 %r33, %r32, 1;
                    setp.lt.u32 %p2, %r33, {t.InnerLimit};
                    cvt.u64.u32 %rd11, %r31;
                    mul.lo.u64 %rd11, %rd11, {leading};
                    mul.wide.u32 %rd12, %r32, {(words ? 2 : 4)};
                    add.u64 %rd11, %rd11, %rd12;
                    add.u64 %rd11, %rd11, {baseAddress};
                    mul.lo.u64 %rd13, {leading}, {t.RowStep};
                """);
            for (int r = 0; r < t.Pairs; r++)
            {
                if (words)
                {
                    // The pair (k, k + 1) is one word; a row's odd last value is paired with the weight's zero padding.
                    s.AppendLine($"""
                            setp.lt.u32 %p3, %r31, {t.OuterLimit};
                            and.pred %p4, %p3, %p1;
                            mov.b32 %gw{r}, 0;
                            @%p4 ld.global.b32 %gw{r}, [%rd11];
                            add.u32 %r31, %r31, {t.RowStep};
                            add.u64 %rd11, %rd11, %rd13;
                        """);
                    continue;
                }

                s.AppendLine($"""
                        setp.lt.u32 %p3, %r31, {t.OuterLimit};
                        and.pred %p4, %p3, %p1;
                        and.pred %p5, %p3, %p2;
                        mov.f32 {staging}{2 * r}, 0f00000000;
                        mov.f32 {staging}{2 * r + 1}, 0f00000000;
                        @%p4 ld.global.f32 {staging}{2 * r}, [%rd11];
                        @%p5 ld.global.f32 {staging}{2 * r + 1}, [%rd11+4];
                        add.u32 %r31, %r31, {t.RowStep};
                        add.u64 %rd11, %rd11, %rd13;
                    """);
            }
        }

        // Registers → shared memory stage %r34 (byte offset 0 or StageBytes), rounded to bfloat16 pairs.
        void Store(TileLoad t, string store, string staging)
        {
            s.AppendLine($"    add.u32 %r35, {store}, %r34;");
            for (int r = 0; r < t.Pairs; r++)
            {
                if (bf16B && t.Name == "b")
                {
                    s.AppendLine($"    st.shared.b32 [%r35+{r * t.RowStep * t.Stride}], %gw{r};");
                    continue;
                }

                s.AppendLine($"""
                        cvt.rn.bf16x2.f32 %r36, {staging}{2 * r + 1}, {staging}{2 * r};
                        st.shared.b32 [%r35+{r * t.RowStep * t.Stride}], %r36;
                    """);
            }
        }

        // Packed B: the words of rows k0 + row (+ wordRows · i), word bn / cpw + %r17; zero past the matrix.
        void LoadPacked()
        {
            s.AppendLine($"""
                    add.u32 %r31, %r30, %r16;
                    shr.u32 %r32, %r10, {(int)Math.Log2(cpw)};
                    add.u32 %r32, %r32, %r17;
                    setp.lt.u32 %p1, %r32, %r59;
                    cvt.u64.u32 %rd11, %r31;
                    mul.lo.u64 %rd11, %rd11, %rd10;
                    mul.wide.u32 %rd12, %r32, 4;
                    add.u64 %rd11, %rd11, %rd12;
                    add.u64 %rd11, %rd11, %rd2;
                    mul.lo.u64 %rd13, %rd10, {wordRows};
                """);
            for (int i = 0; i < wordsPerThread; i++)
            {
                s.AppendLine($"""
                        setp.lt.u32 %p3, %r31, %r3;
                        and.pred %p4, %p3, %p1;
                        mov.b32 %gw{i}, 0;
                        @%p4 ld.global.b32 %gw{i}, [%rd11];
                        add.u32 %r31, %r31, {wordRows};
                        add.u64 %rd11, %rd11, %rd13;
                    """);
            }

            if (packed == 2)
            {
                // The tile's rows share one scale group (k0 is a multiple of 32): 8 scales per word, one word per 16 rows.
                s.AppendLine($$"""
                        shr.u32 %r33, %r30, 5;
                        shl.b32 %r36, %r59, 3;
                        mul.lo.u32 %r33, %r33, %r36;
                        shl.b32 %r36, %r32, 3;
                        add.u32 %r33, %r33, %r36;
                        mul.wide.u32 %rd12, %r33, 4;
                        add.u64 %rd12, %rd12, %rd26;
                        setp.lt.u32 %p3, %r30, %r3;
                        and.pred %p4, %p3, %p1;
                        mov.f32 %gs0, 0f00000000;
                        mov.f32 %gs1, 0f00000000;
                        mov.f32 %gs2, 0f00000000;
                        mov.f32 %gs3, 0f00000000;
                        mov.f32 %gs4, 0f00000000;
                        mov.f32 %gs5, 0f00000000;
                        mov.f32 %gs6, 0f00000000;
                        mov.f32 %gs7, 0f00000000;
                        @%p4 ld.global.v4.f32 {%gs0, %gs1, %gs2, %gs3}, [%rd12];
                        @%p4 ld.global.v4.f32 {%gs4, %gs5, %gs6, %gs7}, [%rd12+16];
                    """);
            }
        }

        void StorePacked()
        {
            s.AppendLine("    add.u32 %r35, %r18, %r34;");
            for (int i = 0; i < wordsPerThread; i++)
            {
                int offset = i * wordRows * WideStride;
                if (packed == 3)
                {
                    s.AppendLine($"    st.shared.b32 [%r35+{offset}], %gw{i};");
                    continue;
                }

                int values = packed == 1 ? 4 : 8, bits = packed == 1 ? 8 : 4;
                for (int j = 0; j < values; j++)
                {
                    s.AppendLine($"""
                            bfe.s32 %r36, %gw{i}, {j * bits}, {bits};
                            cvt.rn.f32.s32 %gx{j}, %r36;
                        """);
                    if (packed == 2)
                    {
                        s.AppendLine($"    mul.f32 %gx{j}, %gx{j}, %gs{j};");
                    }
                }

                s.AppendLine("""
                        cvt.rn.bf16x2.f32 %r36, %gx1, %gx0;
                        cvt.rn.bf16x2.f32 %r37, %gx3, %gx2;
                    """);
                if (packed == 1)
                {
                    s.AppendLine($$"""    st.shared.v2.b32 [%r35+{{offset}}], {%r36, %r37};""");
                }
                else
                {
                    s.AppendLine($$"""
                            cvt.rn.bf16x2.f32 %r38, %gx5, %gx4;
                            cvt.rn.bf16x2.f32 %r57, %gx7, %gx6;
                            st.shared.v4.b32 [%r35+{{offset}}], {%r36, %r37, %r38, %r57};
                        """);
                }
            }
        }

        void LoadB()
        {
            if (packed == 0)
            {
                Load(b, "%rd2", "%rd10", "%r16", "%r17", "%gb");
            }
            else
            {
                LoadPacked();
            }
        }

        void StoreB()
        {
            if (packed == 0)
            {
                Store(b, "%r18", "%gb");
            }
            else
            {
                StorePacked();
            }
        }

        s.AppendLine($"""
                mov.u32 %r30, {(mode == 0 ? "%r62" : "0")};
                mov.u32 %r34, 0;
            """);
        Load(a, "%rd1", "%rd9", "%r13", "%r14", "%ga");
        LoadB();
        Store(a, "%r15", "%ga");
        StoreB();
        s.AppendLine("""
                bar.sync 0;
            KLOOP:
                setp.ge.u32 %p10, %r30, %r3;
                @%p10 bra KEND;
                add.u32 %r37, %r30, 32;
                setp.lt.u32 %p11, %r37, %r3;
                mov.u32 %r30, %r37;
                @!%p11 bra NOLOAD;
            """);
        Load(a, "%rd1", "%rd9", "%r13", "%r14", "%ga");
        LoadB();
        s.AppendLine("""
            NOLOAD:
                add.u32 %r39, %r20, %r34;
                add.u32 %r41, %r21, %r34;
            """);

        // Two k16 steps over the current stage (%r39 / %r41: its A / B lane addresses).
        void Mma()
        {
            for (int kk = 0; kk < 2; kk++)
            {
                for (int mt = 0; mt < mts; mt++)
                {
                    int offset = ta ? kk * 16 * WideStride + mt * 16 * 2 : mt * 16 * NarrowStride + kk * 32;
                    s.AppendLine($"    ldmatrix.sync.aligned.m8n8.x4{(ta ? ".trans" : "")}.shared.b16 {{%fa{4 * mt}, %fa{4 * mt + 1}, %fa{4 * mt + 2}, %fa{4 * mt + 3}}}, [%r39+{offset}];");
                }

                for (int np = 0; np < 2; np++)
                {
                    int offset = tb ? np * 16 * NarrowStride + kk * 32 : kk * 16 * WideStride + np * 16 * 2;
                    s.AppendLine($"    ldmatrix.sync.aligned.m8n8.x4{(tb ? "" : ".trans")}.shared.b16 {{%fb{4 * np}, %fb{4 * np + 1}, %fb{4 * np + 2}, %fb{4 * np + 3}}}, [%r41+{offset}];");
                }

                for (int mt = 0; mt < mts; mt++)
                {
                    for (int nt = 0; nt < 4; nt++)
                    {
                        int c = (mt * 4 + nt) * 4, fb = (nt / 2) * 4 + (nt % 2) * 2;
                        s.AppendLine($"    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {{%c{c}, %c{c + 1}, %c{c + 2}, %c{c + 3}}}, "
                                     + $"{{%fa{4 * mt}, %fa{4 * mt + 1}, %fa{4 * mt + 2}, %fa{4 * mt + 3}}}, {{%fb{fb}, %fb{fb + 1}}}, "
                                     + $"{{%c{c}, %c{c + 1}, %c{c + 2}, %c{c + 3}}};");
                    }
                }
            }
        }

        Mma();

        s.AppendLine($"""
                @!%p11 bra NOSTORE;
                xor.b32 %r34, %r34, {StageBytes};
            """);
        Store(a, "%r15", "%ga");
        StoreB();
        s.AppendLine("""
            NOSTORE:
                bar.sync 0;
                bra KLOOP;
            KEND:
            """);
        if (lowRank)
        {
            // One more k step: u's tile as A (rows of r), v's as B (r × n as stored, or n × r transposed), zero past r.
            var u = a with { Name = "u", InnerLimit = "%lr7" };
            var v = tb ? new TileLoad("v", KIsOuter: false, Width: TensorK, OuterLimit: "%r2", InnerLimit: "%lr7", Tile: "%r10")
                       : new TileLoad("v", KIsOuter: true, Width: TensorTile, OuterLimit: "%lr7", InnerLimit: "%r2", Tile: "%r10");
            s.AppendLine("""
                    setp.eq.u64 %plr, %lrd2, 0;
                    mov.u32 %lr0, %ctaid.z;
                    mov.u32 %lr4, %nctaid.z;
                    setp.gt.u32 %p10, %lr4, 1;
                    setp.ne.and.u32 %p10, %lr0, 0, %p10;
                    or.pred %plr, %plr, %p10;
                    @%plr bra LR_DONE;
                    cvta.to.global.u64 %lrd2, %lrd2;
                    cvta.to.global.u64 %lrd3, %lrd3;
                    ld.param.u32 %lr7, [p_r];
                    mul.wide.u32 %lrd1, %lr7, 4;
                    mov.u32 %r30, 0;
                    mov.u32 %r34, 0;
                """);
            Coordinates(v, "%lr1", "%lr2", "%lr3", "%r12", "%lrd4", tb ? "%lr7" : "%r2");
            Load(u, "%lrd2", "%lrd1", "%r13", "%r14", "%ga");
            Load(v, "%lrd3", "%lrd4", "%lr1", "%lr2", "%gb");
            Store(u, "%r15", "%ga");
            Store(v, "%lr3", "%gb");
            s.AppendLine("""
                    bar.sync 0;
                    mov.u32 %r39, %r20;
                    mov.u32 %r41, %r21;
                """);
            Mma();
            s.AppendLine("LR_DONE:");
        }

        EmitTensorEpilogue(s, mode, packed == 1 ? t => ColumnScales(t, mts) : null, splitK: mode == 0, mts: mts);
        s.AppendLine("""
                ret;
            }
            """);
        if (multi)
        {
            // Every read of the scales (and the epilogue's auxiliary pointer) takes the selected product's.
            foreach (string register in new[] { "%rd21", "%rd24", "%rd25", "%rd26" })
            {
                s.Replace($"ld.param.u64 {register}, [p_aux];", $"mov.u64 {register}, %rdaux;");
            }
        }

        sb.Append(s);
        sb.AppendLine();
    }

    // The epilogue of the tensor-core products (and of the 8-bit ones): c[row, col] = beta·c + f(acc), rows (lane >> 2)
    // and +8 of each m16 tile, columns 2 (lane & 3) and +1 of each n8 tile. %r42 = first row, %r43 = first column,
    // %rd14 = its address, %rd15 = a row in bytes. Expects %r1 = m, %r2 = n, %r5 = lane, %r7 / %r8 = the warp's tile row /
    // column, %r9 / %r10 = the block's first row / column, %r60 = ldc, %rd3 = c, %beta / %pbeta, float accumulators %c0-63,
    // and the parameters p_bias and p_aux. `scale` may rescale the accumulators once %r42 / %r43 are set. With `splitK`
    // (mode 0), %psplit (set by the caller) marks a split k: each block adds its partial sums into c atomically (beta
    // must be 0, c zeroed first, or 1) and only block z = 0 adds the bias.
    // mts: the m16 slices of each warp (4: 64 rows, the 128-row tiles; 2: 32 rows, the 64-row tiles).
    private static void EmitTensorEpilogue(StringBuilder s, int mode, Action<StringBuilder>? scale, bool splitK = false, int mts = 4)
    {
        // Epilogue: c[row, col] = acc + beta · c (rows (lane >> 2) and +8 of each m16 tile, columns 2 (lane & 3) and +1
        // of each n8 tile). %r42 = first row, %r43 = first column, %rd14 = its address, %rd15 = a row in bytes.
        s.AppendLine($"""
                shr.u32 %r42, %r5, 2;
                shl.b32 %r44, %r7, {(int)Math.Log2(mts * 16)};
                add.u32 %r42, %r42, %r44;
                add.u32 %r42, %r42, %r9;
                and.b32 %r43, %r5, 3;
                shl.b32 %r43, %r43, 1;
                shl.b32 %r44, %r8, 5;
                add.u32 %r43, %r43, %r44;
                add.u32 %r43, %r43, %r10;
                mul.wide.u32 %rd15, %r60, 4;
                cvt.u64.u32 %rd14, %r42;
                mul.lo.u64 %rd14, %rd14, %rd15;
                mul.wide.u32 %rd16, %r43, 4;
                add.u64 %rd14, %rd14, %rd16;
                add.u64 %rd14, %rd14, %rd3;
            """);
        // Per row group (8 elements of one row): read the old values first, all at once (pairs as 8-byte loads when n is
        // even, so both columns share an aligned pair), then add and store. Reading them one by one between the stores
        // made the epilogue wait on memory 64 times per thread, halving the speed of beta ≠ 0 products.
        // Bias (optional, one value per column): this thread's 8 columns, loaded once.
        // Pairs are 8-byte aligned when ldc is even and c (and the auxiliary tensor, same layout) start 8-byte aligned.
        // Modes: 0 plain; 1 GELU: aux (if any) = acc + bias, c = beta·c + gelu(acc + bias); 2 GELU gradient:
        // c = beta·c + acc · gelu'(aux) (no bias).
        s.AppendLine("""
                ld.param.u64 %rd21, [p_aux];
                setp.ne.u64 %paux, %rd21, 0;
                cvta.to.global.u64 %rd21, %rd21;
                and.b32 %r47, %r60, 1;
                setp.eq.u32 %peven, %r47, 0;
                cvt.u32.u64 %r47, %rd3;
                ld.param.u64 %rd24, [p_aux];
                cvt.u32.u64 %r48, %rd24;
                or.b32 %r47, %r47, %r48;
                and.b32 %r47, %r47, 7;
                setp.eq.and.u32 %peven, %r47, 0, %peven;
                sub.u64 %rd22, %rd21, %rd3;
                ld.param.u64 %rd18, [p_bias];
                setp.ne.u64 %pbias, %rd18, 0;
                cvta.to.global.u64 %rd18, %rd18;
            """);
        if (splitK)
        {
            s.AppendLine("""
                    @%psplit mov.f32 %beta, 0f00000000;
                    @%psplit setp.ne.u32 %pbeta, %r49, %r49;
                    mov.u32 %r49, %ctaid.z;
                    setp.eq.u32 %pz0, %r49, 0;
                    and.pred %pbias, %pbias, %pz0;
                """);
        }

        for (int nt = 0; nt < 4; nt++)
        {
            for (int j = 0; j < 2; j++)
            {
                s.AppendLine($$"""
                        mov.f32 %bias{{2 * nt + j}}, 0f00000000;
                        add.u32 %r46, %r43, {{nt * 8 + j}};
                        setp.lt.u32 %p9, %r46, %r2;
                        and.pred %p9, %p9, %pbias;
                        mul.wide.u32 %rd20, %r46, 4;
                        add.u64 %rd19, %rd18, %rd20;
                        @%p9 ld.global.f32 %bias{{2 * nt + j}}, [%rd19];
                    """);
            }
        }

        scale?.Invoke(s);
        for (int mt = 0; mt < mts; mt++)
        {
            for (int half = 0; half < 2; half++)
            {
                int dr = mt * 16 + half * 8;
                s.AppendLine($"""
                        add.u32 %r45, %r42, {dr};
                        setp.lt.u32 %p6, %r45, %r1;
                        mul.lo.u64 %rd17, %rd15, {dr};
                        add.u64 %rd17, %rd17, %rd14;
                    """);
                for (int nt = 0; nt < 4; nt++)
                {
                    // %q(3nt): the pair as one vector; %q(3nt+1), %q(3nt+2): single columns (odd n, or the last column).
                    s.AppendLine($"""
                            add.u32 %r46, %r43, {nt * 8};
                            setp.lt.u32 %q{3 * nt + 1}, %r46, %r2;
                            and.pred %q{3 * nt + 1}, %q{3 * nt + 1}, %p6;
                            add.u32 %r46, %r46, 1;
                            setp.lt.u32 %q{3 * nt + 2}, %r46, %r2;
                            and.pred %q{3 * nt + 2}, %q{3 * nt + 2}, %p6;
                            and.pred %q{3 * nt}, %q{3 * nt + 2}, %peven;
                            not.pred %qnot, %q{3 * nt};
                            and.pred %q{3 * nt + 1}, %q{3 * nt + 1}, %qnot;
                            and.pred %q{3 * nt + 2}, %q{3 * nt + 2}, %qnot;
                            mov.f32 %e{2 * nt}, 0f00000000;
                            mov.f32 %e{2 * nt + 1}, 0f00000000;
                        """);
                }

                int group = mt * 2 + half;
                s.AppendLine($"@!%pbeta bra EPI_AUX_{group};");
                for (int nt = 0; nt < 4; nt++)
                {
                    s.AppendLine($$"""
                            @%q{{3 * nt}} ld.global.v2.f32 {%e{{2 * nt}}, %e{{2 * nt + 1}}}, [%rd17+{{nt * 32}}];
                            @%q{{3 * nt + 1}} ld.global.f32 %e{{2 * nt}}, [%rd17+{{nt * 32}}];
                            @%q{{3 * nt + 2}} ld.global.f32 %e{{2 * nt + 1}}, [%rd17+{{nt * 32 + 4}}];
                        """);
                }

                s.AppendLine($"EPI_AUX_{group}:");
                s.AppendLine("add.u64 %rd23, %rd17, %rd22;");
                if (mode == 2)
                {
                    // The pre-activations (same layout as c, at c + %rd22).
                    for (int nt = 0; nt < 4; nt++)
                    {
                        s.AppendLine($$"""
                                @%q{{3 * nt}} ld.global.v2.f32 {%h{{2 * nt}}, %h{{2 * nt + 1}}}, [%rd23+{{nt * 32}}];
                                @%q{{3 * nt + 1}} ld.global.f32 %h{{2 * nt}}, [%rd23+{{nt * 32}}];
                                @%q{{3 * nt + 2}} ld.global.f32 %h{{2 * nt + 1}}, [%rd23+{{nt * 32 + 4}}];
                            """);
                    }
                }

                // Mode 0.
                for (int nt = 0; nt < 4 && mode == 0; nt++)
                {
                    int c = (mt * 4 + nt) * 4 + half * 2;
                    s.AppendLine($$"""
                            fma.rn.f32 %e{{2 * nt}}, %e{{2 * nt}}, %beta, %c{{c}};
                            fma.rn.f32 %e{{2 * nt + 1}}, %e{{2 * nt + 1}}, %beta, %c{{c + 1}};
                            add.f32 %e{{2 * nt}}, %e{{2 * nt}}, %bias{{2 * nt}};
                            add.f32 %e{{2 * nt + 1}}, %e{{2 * nt + 1}}, %bias{{2 * nt + 1}};
                        """);
                }

                // Mode 1: pre = acc + bias (stored when aux is given), c = beta·c + gelu(pre).
                for (int nt = 0; nt < 4 && mode == 1; nt++)
                {
                    int c = (mt * 4 + nt) * 4 + half * 2;
                    s.AppendLine($$"""
                            add.f32 %h{{2 * nt}}, %c{{c}}, %bias{{2 * nt}};
                            add.f32 %h{{2 * nt + 1}}, %c{{c + 1}}, %bias{{2 * nt + 1}};
                            and.pred %p12, %q{{3 * nt}}, %paux;
                            and.pred %p13, %q{{3 * nt + 1}}, %paux;
                            and.pred %p14, %q{{3 * nt + 2}}, %paux;
                            @%p12 st.global.v2.f32 [%rd23+{{nt * 32}}], {%h{{2 * nt}}, %h{{2 * nt + 1}}};
                            @%p13 st.global.f32 [%rd23+{{nt * 32}}], %h{{2 * nt}};
                            @%p14 st.global.f32 [%rd23+{{nt * 32 + 4}}], %h{{2 * nt + 1}};
                        """);
                    for (int j = 0; j < 2; j++)
                    {
                        s.AppendLine(GeluTerms($"%h{2 * nt + j}"));
                        s.AppendLine($"""
                                add.f32 %t6, %t5, 0f3F800000;
                                mul.f32 %t6, %t6, %h{2 * nt + j};
                                mul.f32 %t6, %t6, 0f3F000000;
                                fma.rn.f32 %e{2 * nt + j}, %e{2 * nt + j}, %beta, %t6;
                            """);
                    }
                }

                // Mode 2: c = beta·c + acc · gelu'(pre).
                for (int nt = 0; nt < 4 && mode == 2; nt++)
                {
                    int c = (mt * 4 + nt) * 4 + half * 2;
                    for (int j = 0; j < 2; j++)
                    {
                        string x = $"%h{2 * nt + j}";
                        s.AppendLine(GeluTerms(x));
                        s.AppendLine($"""
                                add.f32 %t6, %t5, 0f3F800000;
                                mul.f32 %t6, %t6, 0f3F000000;
                                mul.f32 %t7, %t5, %t5;
                                sub.f32 %t7, 0f3F800000, %t7;
                                mul.f32 %t0, %t0, {F(3f * 0.044715f)};
                                add.f32 %t0, %t0, 0f3F800000;
                                mul.f32 %t0, %t0, {F(0.7978845608f)};
                                mul.f32 %t7, %t7, %t0;
                                mul.f32 %t7, %t7, {x};
                                fma.rn.f32 %t6, %t7, 0f3F000000, %t6;
                                mul.f32 %t6, %t6, %c{c + j};
                                fma.rn.f32 %e{2 * nt + j}, %e{2 * nt + j}, %beta, %t6;
                            """);
                    }
                }

                if (splitK)
                {
                    s.AppendLine($"@!%psplit bra EPI_STORE_{group};");
                    for (int nt = 0; nt < 4; nt++)
                    {
                        s.AppendLine($"""
                                or.pred %p12, %q{3 * nt}, %q{3 * nt + 1};
                                or.pred %p13, %q{3 * nt}, %q{3 * nt + 2};
                                @%p12 red.global.add.f32 [%rd17+{nt * 32}], %e{2 * nt};
                                @%p13 red.global.add.f32 [%rd17+{nt * 32 + 4}], %e{2 * nt + 1};
                            """);
                    }

                    s.AppendLine($"bra EPI_DONE_{group};");
                    s.AppendLine($"EPI_STORE_{group}:");
                }

                for (int nt = 0; nt < 4; nt++)
                {
                    s.AppendLine($$"""
                            @%q{{3 * nt}} st.global.v2.f32 [%rd17+{{nt * 32}}], {%e{{2 * nt}}, %e{{2 * nt + 1}}};
                            @%q{{3 * nt + 1}} st.global.f32 [%rd17+{{nt * 32}}], %e{{2 * nt}};
                            @%q{{3 * nt + 2}} st.global.f32 [%rd17+{{nt * 32 + 4}}], %e{{2 * nt + 1}};
                        """);
                }

                if (splitK)
                {
                    s.AppendLine($"EPI_DONE_{group}:");
                }
            }
        }
    }

    // Int8 weights: acc · scale[column] (the per-column scales in p_aux).
    private static void ColumnScales(StringBuilder s, int mts)
    {
        s.AppendLine("""
                ld.param.u64 %rd25, [p_aux];
                cvta.to.global.u64 %rd25, %rd25;
            """);
        for (int nt = 0; nt < 4; nt++)
        {
            for (int j = 0; j < 2; j++)
            {
                s.AppendLine($$"""
                        mov.f32 %t{{0}}, 0f3F800000;
                        add.u32 %r46, %r43, {{nt * 8 + j}};
                        setp.lt.u32 %p9, %r46, %r2;
                        mul.wide.u32 %rd20, %r46, 4;
                        add.u64 %rd19, %rd25, %rd20;
                        @%p9 ld.global.f32 %t0, [%rd19];
                    """);
                for (int mt = 0; mt < mts; mt++)
                {
                    int c = (mt * 4 + nt) * 4 + j;
                    s.AppendLine($"mul.f32 %c{c}, %c{c}, %t0;");
                    s.AppendLine($"mul.f32 %c{c + 2}, %c{c + 2}, %t0;");
                }
            }
        }
    }
}
