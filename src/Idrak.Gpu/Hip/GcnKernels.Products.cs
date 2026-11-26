// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;

namespace Idrak.Gpu.Hip;

// HIP's matrix products in GCN (plan 35, G4a), written for gfx9's wave of 64 as CUDA's are in PTX (Cuda/PtxKernels.cs,
// .Rows.cs): CUDA's names, CUDA's arguments in CUDA's order, the CPU backend's results within the
// device-against-CPU tolerance. Every output's sum is taken in one order fixed by the shapes (rule 86): over k in
// increasing order within a thread, then the parts (slices of k inside a block, splits of k across blocks) added in
// their order, by LDS in a fixed shape, by the launch's last block or by a second kernel, never added by atomics; so a
// result is the same on every grid,
// every wave order and every device of the family.
//
// - matmul_f32: c[b] = alpha · op(a[b]) · op(b[b]) + beta · c[b] (+ bias[j] on every row), CUDA's matmul_f32 (its twelve
//   arguments, then alpha and bias, 0 for none). Tiles of 64 × 64 outputs, blocks of 256 (a 4 × 4 tile a thread), k in
//   steps of 16 through LDS (A as 16 × 64 and B as 16 × 64, k-major, so a thread reads its four rows and its four
//   columns with one ds_read_b128 each). The global reads are coalesced for every transposition: along k where the
//   operand is stored along k (A as stored, B transposed), along the rows or columns otherwise; each load's place in its
//   tile, its offset, its LDS slot and whether its row exists are worked out once at the start (rule 85), out-of-range
//   elements read as zero. Grid: x = ⌈n / 64⌉, y = ⌈m / 64⌉, z = batch (strides in elements).
// - gemv_nn_f32: few rows (decoding) through b [k, n], 8 rows a block (rows past m read row m - 1, not stored); 64
//   columns × 4 slices of k (a wave each, k in four contiguous slices of ⌈k / 16⌉ · 4), four k a step with four loads of
//   b in flight and the inputs a[r, k..k+3] read by the scalar unit (one load per row, shared by the wave); the slices'
//   sums added in LDS in slice order. Grid: x = ⌈n / 64⌉, y = ⌈m / 8⌉, z = batch.
// - gemv_nt_f32: few rows through b stored [n, k] (a product with a transposed matrix): a wave per column, lanes
//   striding over k (coalesced, 4 · 64 k a step, 36 loads in flight), then each row's 64 lane sums by one tree of a fixed
//   shape in the wave's LDS. Grid: x = ⌈n / 4⌉, y = ⌈m / 8⌉, z = batch.
// - gemv_multi_f32: CUDA's float32 multi-product (Backend.MatMulMany: the query, key and value projections, or gate
//   and up, of one input): up to three products y_j = a · w_j (+ bias_j) of a [m ≤ 8, k], gemv_nn_f32's body with grid
//   y picking the product (its arguments selected once) and grid x the widest product's columns. Grid: x = ⌈max n_j / 64⌉,
//   y = the products.
//
// Bounds: every matrix below 2^30 elements (byte offsets are 32-bit), grids below 2^29 work-items (grid-stride indexes
// do not wrap); the backend bounds its grids by the device's limits (HipDeviceLimits) and splits larger work.
//
// For --profile labels (CUDA's "{kind}_{nn} {m}x{n}x{k}" with 2·m·n·k operations) and --bench-hip rows: each kernel's
// floating-point operations (a fused multiply-add counts 2) and global-memory bytes, as formulas of its shape; b = batch,
// s = splits.
//
// | Kernel                 | Arguments                                                        | Launch (block)                  | Flops            | Bytes                                         |
// |------------------------|------------------------------------------------------------------|---------------------------------|------------------|-----------------------------------------------|
// | matmul_f32             | a, b, c, m, n, k, ta, tb, beta, sa, sb, sc, alpha, bias          | ⌈n/64⌉ × ⌈m/64⌉ × b (256)       | 2·m·n·k·b        | 4·b·(m·k + k·n + m·n·(beta ≠ 0 ? 2 : 1)) + 4n |
// | gemv_nn_f32            | a, b, c, m, n, k, beta, sa, sb, sc                               | ⌈n/64⌉ × ⌈m/8⌉ × b (256)        | 2·m·n·k·b        | 4·b·(m·k + ⌈m/8⌉·k·n + m·n)                   |
// | gemv_nt_f32            | a, b, c, m, n, k, beta, sa, sb, sc                               | ⌈n/4⌉ × ⌈m/8⌉ × b (256)         | 2·m·n·k·b        | 4·b·(m·k + ⌈m/8⌉·k·n + m·n)                   |
// | gemv_multi_f32         | a, m, k; 3 × (w, bias, y, n)                                     | ⌈max n/64⌉ × products (256)     | 2·m·n·k          | 4·(m·k + k·n + m·n + n)                       |
//
// (n in a row's Bytes is that kernel's own argument: the output columns; for gemv_multi_f32, the products' columns
// together.)
internal static partial class GcnKernels
{
    /// <summary>The products and the few-row products, in the order they join <see cref="All"/>.</summary>
    internal static IReadOnlyList<GcnKernelSource> Products => _products ??= BuildProducts();

    // Built on first use: the text reads this file's argument lists and names, whose static initializers come later.
    private static GcnKernelSource[]? _products;

    /// <summary>Outputs a <c>matmul_f32</c> block takes in each direction (its tile is 64 × 64).</summary>
    internal const int MatMulTile = 64;

    /// <summary>k a <c>matmul_f32</c> tile step takes through LDS.</summary>
    internal const int MatMulDepth = 16;

    /// <summary>Threads of a <c>matmul_f32</c> block: 16 × 16, each a 4 × 4 tile of outputs.</summary>
    internal const int MatMulThreads = 256;

    /// <summary>Rows of a few-row float32 product's block (<c>gemv_nn_f32</c>, <c>gemv_nt_f32</c>), as CUDA's GemvRows.</summary>
    internal const int GemvRows = 8;

    /// <summary>Threads of a <c>gemv_nn_f32</c> or <c>gemv_nt_f32</c> block (four waves).</summary>
    internal const int GemvThreads = 256;

    /// <summary>Columns a <c>gemv_nn_f32</c> block takes (a lane each; its four waves split k).</summary>
    internal const int GemvNnColumns = 64;

    /// <summary>Columns a <c>gemv_nt_f32</c> block takes (a wave each).</summary>
    internal const int GemvNtColumns = 4;

    /// <summary>The block of the grid-stride kernels.</summary>
    internal const int ExpansionBlock = 256;

    // ---- Arguments and packing -------------------------------------------------------------------------------------

    /// <summary>
    /// matmul_f32's arguments: CUDA's (a, b, c at 0, 8, 16; m, n, k, ta, tb at 24-40; beta at 44; the batch strides sa,
    /// sb, sc in elements at 48-56), then alpha at 60 and bias at 64 (0: none).
    /// </summary>
    internal static IReadOnlyList<GcnArgument> MatMulArguments { get; } =
    [
        GcnArgument.Buffer(0, "a"), GcnArgument.Buffer(8, "b"), GcnArgument.Buffer(16, "c"), GcnArgument.Value(24, 4, "m"), GcnArgument.Value(28, 4, "n"),
        GcnArgument.Value(32, 4, "k"), GcnArgument.Value(36, 4, "ta"), GcnArgument.Value(40, 4, "tb"), GcnArgument.Value(44, 4, "beta"),
        GcnArgument.Value(48, 4, "sa"), GcnArgument.Value(52, 4, "sb"), GcnArgument.Value(56, 4, "sc"), GcnArgument.Value(60, 4, "alpha"),
        GcnArgument.Buffer(64, "bias"),
    ];

    /// <summary>Bytes of matmul_f32's arguments.</summary>
    internal const int MatMulArgumentBytes = 72;

    /// <summary>gemv_nn_f32's and gemv_nt_f32's arguments, CUDA's: a, b, c; m, n, k, beta; the batch strides sa, sb, sc (64-bit, in elements).</summary>
    internal static IReadOnlyList<GcnArgument> GemvArguments { get; } =
    [
        GcnArgument.Buffer(0, "a"), GcnArgument.Buffer(8, "b"), GcnArgument.Buffer(16, "c"), GcnArgument.Value(24, 4, "m"), GcnArgument.Value(28, 4, "n"),
        GcnArgument.Value(32, 4, "k"), GcnArgument.Value(36, 4, "beta"), GcnArgument.Value(40, 8, "sa"), GcnArgument.Value(48, 8, "sb"),
        GcnArgument.Value(56, 8, "sc"),
    ];

    /// <summary>Bytes of a few-row float32 product's arguments.</summary>
    internal const int GemvArgumentBytes = 64;

    /// <summary>The most products <c>gemv_multi_f32</c> takes (grid y), as CUDA's.</summary>
    internal const int GemvMultiProducts = 3;

    /// <summary>
    /// gemv_multi_f32's arguments, CUDA's: a; m, k; then for products 0 to 2 the weights w [k, n], the bias (0: none),
    /// the output y and the columns n, from 16 + 32j.
    /// </summary>
    internal static IReadOnlyList<GcnArgument> GemvMultiArguments { get; } =
    [
        GcnArgument.Buffer(0, "a"), GcnArgument.Value(8, 4, "m"), GcnArgument.Value(12, 4, "k"),
        GcnArgument.Buffer(16, "w0"), GcnArgument.Buffer(24, "bias0"), GcnArgument.Buffer(32, "y0"), GcnArgument.Value(40, 4, "n0"),
        GcnArgument.Buffer(48, "w1"), GcnArgument.Buffer(56, "bias1"), GcnArgument.Buffer(64, "y1"), GcnArgument.Value(72, 4, "n1"),
        GcnArgument.Buffer(80, "w2"), GcnArgument.Buffer(88, "bias2"), GcnArgument.Buffer(96, "y2"), GcnArgument.Value(104, 4, "n2"),
    ];

    /// <summary>Bytes of gemv_multi_f32's arguments (the last product's columns end them).</summary>
    internal const int GemvMultiArgumentBytes = 108;

    /// <summary>One product of <c>gemv_multi_f32</c>: the weights [k, n], the bias (0: none), the output and the columns.</summary>
    internal readonly record struct FloatProduct(ulong Weights, ulong Bias, ulong Output, uint Columns);

    /// <summary>
    /// gemv_multi_f32's arguments (<see cref="GemvMultiArguments"/>) into <paramref name="arguments"/>, <see cref="GemvMultiArgumentBytes"/>
    /// long; products past those given are the last again with 0 columns (their blocks return at once), as CUDA's launch.
    /// </summary>
    internal static void PackGemvMulti(Span<byte> arguments, ulong a, uint m, uint k, ReadOnlySpan<FloatProduct> products)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(arguments.Length, GemvMultiArgumentBytes);
        ArgumentOutOfRangeException.ThrowIfLessThan(products.Length, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(products.Length, GemvMultiProducts);
        Span<ulong> values = stackalloc ulong[3 + 4 * GemvMultiProducts];
        (values[0], values[1], values[2]) = (a, m, k);
        for (int j = 0; j < GemvMultiProducts; j++)
        {
            var p = products[Math.Min(j, products.Length - 1)];
            values[3 + 4 * j] = p.Weights;
            values[4 + 4 * j] = p.Bias;
            values[5 + 4 * j] = p.Output;
            values[6 + 4 * j] = j < products.Length ? p.Columns : 0u;
        }

        PackArguments(GemvMultiArguments, arguments, values);
    }

    /// <summary>matmul_f32's arguments (<see cref="MatMulArguments"/>) into <paramref name="arguments"/>, <see cref="MatMulArgumentBytes"/> long.</summary>
    internal static void PackMatMul(Span<byte> arguments, ulong a, ulong b, ulong c, uint m, uint n, uint k, bool transA, bool transB, float beta,
        uint strideA, uint strideB, uint strideC, float alpha = 1f, ulong bias = 0)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(arguments.Length, MatMulArgumentBytes);
        PackArguments(MatMulArguments, arguments, a, b, c, m, n, k, transA ? 1u : 0u, transB ? 1u : 0u, BitConverter.SingleToUInt32Bits(beta), strideA, strideB,
            strideC, BitConverter.SingleToUInt32Bits(alpha), bias);
    }

    /// <summary>gemv_nn_f32's or gemv_nt_f32's arguments (<see cref="GemvArguments"/>) into <paramref name="arguments"/>, <see cref="GemvArgumentBytes"/> long.</summary>
    internal static void PackGemv(Span<byte> arguments, ulong a, ulong b, ulong c, uint m, uint n, uint k, float beta, ulong strideA, ulong strideB, ulong strideC)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(arguments.Length, GemvArgumentBytes);
        PackArguments(GemvArguments, arguments, a, b, c, m, n, k, BitConverter.SingleToUInt32Bits(beta), strideA, strideB, strideC);
    }

    private static GcnKernelSource[] BuildProducts()
    {
        return [MatMul(), GemvNN(), GemvNT(), GemvNN(multi: true)];
    }

    // ---- matmul_f32 ------------------------------------------------------------------------------------------------

    // The registers of one operand's tile loads: per load, its byte offset in the operand, its k index (with bit 31 set
    // when its row or column does not exist, so it never reads), and its LDS slot.
    private sealed record TileLoads(string[] Offsets, string[] Ks, string[] Slots, string Step);

    private static GcnKernelSource MatMul()
    {
        var w = new ProductWriter("matmul_f32");
        w.Head(MatMulArguments, MatMulThreads, lds: 2 * MatMulDepth * MatMulTile * 4, workgroupIds: 3);
        string m = w.Scalar("m"), n = w.Scalar("n"), k = w.Scalar("k"), beta = w.Scalar("beta"), alpha = w.Scalar("alpha"), bias = w.Pointer("bias");
        string aBase = w.Strided(w.Pointer("a"), w.GroupZ, w.Scalar("sa"));
        string bBase = w.Strided(w.Pointer("b"), w.GroupZ, w.Scalar("sb"));
        string cBase = w.Strided(w.Pointer("c"), w.GroupZ, w.Scalar("sc"));
        string row0 = w.S(), col0 = w.S();
        w.L($"s_lshl_b32 {row0}, {w.GroupY}, 6                ; the tile's first row and column");
        w.L($"s_lshl_b32 {col0}, {w.GroupX}, 6");

        // A as stored ([m, k]) is read along k, transposed ([k, m]) along m; B as stored ([k, n]) along n, transposed
        // ([n, k]) along k. Each load's place is set once here.
        var la = TileMap(w, w.Scalar("ta"), alongIndexWhenSet: true, m, k, row0);
        var lb = TileMap(w, w.Scalar("tb"), alongIndexWhenSet: false, n, k, col0);

        string[] acc = new string[16];
        for (int i = 0; i < acc.Length; i++)
        {
            acc[i] = w.V();
            w.L($"v_mov_b32_e32 {acc[i]}, 0");
        }

        string tx = w.V(), ty = w.V(), aRead = w.V(), bRead = w.V();
        w.L($"v_and_b32_e32 {tx}, 15, {w.Tid}               ; this thread's four columns (tx) and four rows (ty)");
        w.L($"v_lshrrev_b32_e32 {ty}, 4, {w.Tid}");
        w.L($"v_lshlrev_b32_e32 {aRead}, 4, {ty}");
        w.L($"v_lshlrev_b32_e32 {bRead}, 4, {tx}");
        string t0 = w.S();
        w.L($"s_mov_b32 {t0}, 0");
        w.Label(".Ltile");
        w.L($"s_cmp_ge_u32 {t0}, {k}");
        w.L($"s_cbranch_scc1 .Lstore");
        string[] masks = new string[8], values = new string[8];
        for (int q = 0; q < 8; q++)
        {
            var l = q < 4 ? la : lb;
            string address = w.V();
            masks[q] = w.SPair();
            values[q] = w.V();
            w.L($"v_cmp_gt_u32_e64 {masks[q]}, {k}, {l.Ks[q % 4]}");
            w.L($"v_cndmask_b32_e64 {address}, 0, {l.Offsets[q % 4]}, {masks[q]}");
            w.L($"global_load_dword {values[q]}, {address}, {(q < 4 ? aBase : bBase)}");
        }

        for (int q = 0; q < 8; q++)
        {
            var l = q < 4 ? la : lb;
            w.L($"v_cndmask_b32_e64 {values[q]}, 0, {values[q]}, {masks[q]}");
            w.L($"ds_write_b32 {l.Slots[q % 4]}, {values[q]}{(q < 4 ? "" : " offset:4096")}");
        }

        w.L($"s_barrier");
        for (int group = 0; group < MatMulDepth / 4; group++)
        {
            // Each group of four k its own block: its reads in flight together (rule 80), their registers reused by the next.
            w.Label(string.Create(CultureInfo.InvariantCulture, $".Lk{group}"));
            int[] ra = new int[4], rb = new int[4];
            for (int u = 0; u < 4; u++)
            {
                int kk = 4 * group + u;
                ra[u] = w.Range(4);
                rb[u] = w.Range(4);
                w.L($"ds_read_b128 %v[{ra[u]}:{ra[u] + 3}], {aRead}{ProductWriter.Offset(256 * kk)}");
                w.L($"ds_read_b128 %v[{rb[u]}:{rb[u] + 3}], {bRead}{ProductWriter.Offset(4096 + 256 * kk)}");
            }

            for (int u = 0; u < 4; u++)
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        w.L($"v_fma_f32 {acc[4 * i + j]}, %v{ra[u] + i}, %v{rb[u] + j}, {acc[4 * i + j]}");
                    }
                }
            }
        }

        w.L($"s_barrier");
        for (int q = 0; q < 8; q++)
        {
            var l = q < 4 ? la : lb;
            w.L($"v_add_u32_e32 {l.Offsets[q % 4]}, {l.Step}, {l.Offsets[q % 4]}");
            w.L($"v_add_u32_e32 {l.Ks[q % 4]}, 16, {l.Ks[q % 4]}");
        }

        w.L($"s_add_u32 {t0}, {t0}, 16");
        w.L($"s_branch .Ltile");

        // c = alpha · Σ, then + beta · c (unless beta is ±0: c is not read, as CUDA's), then + bias[j].
        w.Label(".Lstore");
        string full = w.SPair(), first = w.V();
        w.L($"s_mov_b64 {full}, exec");
        w.L($"v_lshlrev_b32_e32 {first}, 2, {ty}");
        w.L($"v_add_u32_e32 {first}, {row0}, {first}");
        string[] rowMask = new string[4], rowOff = new string[4], cols = new string[4], colMask = new string[4];
        for (int i = 0; i < 4; i++)
        {
            string row = w.V();
            rowMask[i] = w.SPair();
            rowOff[i] = w.V();
            w.L($"v_add_u32_e32 {row}, {i}, {first}");
            w.L($"v_cmp_gt_u32_e64 {rowMask[i]}, {m}, {row}");
            w.L($"v_mul_lo_u32 {rowOff[i]}, {row}, {n}");
        }

        string firstCol = w.V();
        w.L($"v_lshlrev_b32_e32 {firstCol}, 2, {tx}");
        w.L($"v_add_u32_e32 {firstCol}, {col0}, {firstCol}");
        for (int j = 0; j < 4; j++)
        {
            cols[j] = w.V();
            colMask[j] = w.SPair();
            w.L($"v_add_u32_e32 {cols[j]}, {j}, {firstCol}");
            w.L($"v_cmp_gt_u32_e64 {colMask[j]}, {n}, {cols[j]}");
        }

        string[] offsets = new string[16];
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                offsets[4 * i + j] = w.V();
                w.L($"v_add_u32_e32 {offsets[4 * i + j]}, {rowOff[i]}, {cols[j]}");
                w.L($"v_lshlrev_b32_e32 {offsets[4 * i + j]}, 2, {offsets[4 * i + j]}");
                w.L($"v_mul_f32_e32 {acc[4 * i + j]}, {alpha}, {acc[4 * i + j]}");
            }
        }

        string magnitude = w.S();
        w.L($"s_and_b32 {magnitude}, {beta}, 0x7fffffff");
        w.L($"s_cmp_eq_u32 {magnitude}, 0");
        w.L($"s_cbranch_scc1 .Lno_beta");
        // Each output's old value read by its own lane alone (another wave may be writing the tile's first element).
        string[] old = new string[16];
        for (int e = 0; e < 16; e++)
        {
            old[e] = w.V();
            w.L($"v_mov_b32_e32 {old[e]}, 0");
        }

        for (int e = 0; e < 16; e++)
        {
            w.L($"s_and_b64 exec, {rowMask[e / 4]}, {colMask[e % 4]}");
            w.L($"global_load_dword {old[e]}, {offsets[e]}, {cBase}");
        }

        w.L($"s_mov_b64 exec, {full}");
        for (int e = 0; e < 16; e++)
        {
            w.L($"v_fma_f32 {acc[e]}, {beta}, {old[e]}, {acc[e]}");
        }

        w.Label(".Lno_beta");
        w.L($"s_cmp_eq_u64 {bias}, 0");
        w.L($"s_cbranch_scc1 .Lno_bias");
        string[] biases = new string[4];
        for (int j = 0; j < 4; j++)
        {
            string address = w.V();
            biases[j] = w.V();
            w.L($"v_lshlrev_b32_e32 {address}, 2, {cols[j]}");
            w.L($"v_cndmask_b32_e64 {address}, 0, {address}, {colMask[j]}");
            w.L($"global_load_dword {biases[j]}, {address}, {bias}");
        }

        for (int e = 0; e < 16; e++)
        {
            w.L($"v_add_f32_e32 {acc[e]}, {acc[e]}, {biases[e % 4]}");
        }

        w.Label(".Lno_bias");
        for (int e = 0; e < 16; e++)
        {
            w.L($"s_and_b64 exec, {rowMask[e / 4]}, {colMask[e % 4]}");
            w.L($"global_store_dword {offsets[e]}, {acc[e]}, {cBase}");
        }

        w.L($"s_mov_b64 exec, {full}");
        w.L($"s_endpgm");
        return w.Source();
    }

    // Where each of a thread's four loads of one operand's 64 × 16 tile goes (see the file's remarks). `bound` is the
    // operand's rows (A: m) or columns (B: n), `first` the tile's first; with the flag set (A transposed) or clear (B as
    // stored) consecutive lanes take consecutive rows or columns (the operand is stored along them), otherwise
    // consecutive k.
    private static TileLoads TileMap(ProductWriter w, string flag, bool alongIndexWhenSet, string bound, string k, string first)
    {
        string[] offsets = [w.V(), w.V(), w.V(), w.V()], ks = [w.V(), w.V(), w.V(), w.V()], slots = [w.V(), w.V(), w.V(), w.V()];
        string step = w.S(), label = w.NewLabel();
        w.L($"s_cmp_lg_u32 {flag}, 0");
        w.L($"s_cbranch_scc{(alongIndexWhenSet ? 1 : 0)} {label}_index");

        // Along k: lanes 16 apart take the next row; k index = tid mod 16, row = tid / 16 + 16q.
        string kk = w.V(), i0 = w.V();
        w.L($"v_and_b32_e32 {kk}, 15, {w.Tid}");
        w.L($"v_lshrrev_b32_e32 {i0}, 4, {w.Tid}");
        for (int q = 0; q < 4; q++)
        {
            string i = w.V(), index = w.V(), t = w.V(), invalid = w.V();
            w.L($"v_add_u32_e32 {i}, {16 * q}, {i0}");
            w.L($"v_add_u32_e32 {index}, {first}, {i}");
            w.L($"v_mul_lo_u32 {t}, {index}, {k}");
            w.L($"v_add_u32_e32 {t}, {t}, {kk}");
            w.L($"v_lshlrev_b32_e32 {offsets[q]}, 2, {t}");
            w.L($"v_cmp_le_u32_e32 vcc, {bound}, {index}");
            w.L($"v_cndmask_b32_e64 {invalid}, 0, 1, vcc");
            w.L($"v_lshl_or_b32 {ks[q]}, {invalid}, 31, {kk}");
            w.L($"v_lshl_add_u32 {t}, {kk}, 6, {i}");
            w.L($"v_lshlrev_b32_e32 {slots[q]}, 2, {t}");
        }

        w.L($"s_mov_b32 {step}, 64");
        w.L($"s_branch {label}_done");

        // Along the rows (or columns): lane = tid mod 64 takes the row, k index = tid / 64 + 4q.
        w.Label(label + "_index");
        string ii = w.V(), k0 = w.V(), idx = w.V(), bad = w.V(), high = w.V();
        w.L($"v_and_b32_e32 {ii}, 63, {w.Tid}");
        w.L($"v_lshrrev_b32_e32 {k0}, 6, {w.Tid}");
        w.L($"v_add_u32_e32 {idx}, {first}, {ii}");
        w.L($"v_cmp_le_u32_e32 vcc, {bound}, {idx}");
        w.L($"v_cndmask_b32_e64 {bad}, 0, 1, vcc");
        w.L($"v_lshlrev_b32_e32 {high}, 31, {bad}");
        for (int q = 0; q < 4; q++)
        {
            string kq = w.V(), t = w.V();
            w.L($"v_add_u32_e32 {kq}, {4 * q}, {k0}");
            w.L($"v_mul_lo_u32 {t}, {kq}, {bound}");
            w.L($"v_add_u32_e32 {t}, {t}, {idx}");
            w.L($"v_lshlrev_b32_e32 {offsets[q]}, 2, {t}");
            w.L($"v_or_b32_e32 {ks[q]}, {high}, {kq}");
            w.L($"v_lshl_add_u32 {t}, {kq}, 6, {ii}");
            w.L($"v_lshlrev_b32_e32 {slots[q]}, 2, {t}");
        }

        w.L($"s_lshl_b32 {step}, {bound}, 6");
        w.Label(label + "_done");
        return new TileLoads(offsets, ks, slots, step);
    }

    // ---- gemv_nn_f32 -----------------------------------------------------------------------------------------------

    // gemv_nn_f32; with multi, gemv_multi_f32 (CUDA's MatMulManyKernel): up to three products of the input a [m ≤ 8, k],
    // y_j = a · w_j (+ bias_j, after the sum, as the CPU's product then AddRowVector) with w_j [k, n_j], grid y picking j
    // (its weights, bias, output and width selected once at the start) and grid x the widest product's columns (the
    // blocks past n_j end at once), one row group; otherwise gemv_nn_f32's body, its order of summation the same.
    private static GcnKernelSource GemvNN(bool multi = false)
    {
        var w = new ProductWriter(multi ? "gemv_multi_f32" : "gemv_nn_f32");
        w.Head(multi ? GemvMultiArguments : GemvArguments, GemvThreads, lds: 4 * GemvRows * GemvNnColumns * 4, workgroupIds: multi ? 2 : 3);
        string m = w.Scalar("m"), k = w.Scalar("k"), n, beta, aBase, bBase, cBase, bias = "";
        if (multi)
        {
            (n, beta, aBase, bBase, cBase, bias) = (w.S(), w.S(), w.Pointer("a"), w.SPair(), w.SPair(), w.SPair());
            w.L($"s_cmp_eq_u32 {w.GroupY}, 1                       ; this row of blocks' product");
            w.L($"s_cselect_b64 {bBase}, {w.Pointer("w1")}, {w.Pointer("w0")}");
            w.L($"s_cselect_b64 {bias}, {w.Pointer("bias1")}, {w.Pointer("bias0")}");
            w.L($"s_cselect_b64 {cBase}, {w.Pointer("y1")}, {w.Pointer("y0")}");
            w.L($"s_cselect_b32 {n}, {w.Scalar("n1")}, {w.Scalar("n0")}");
            w.L($"s_cmp_eq_u32 {w.GroupY}, 2");
            w.L($"s_cselect_b64 {bBase}, {w.Pointer("w2")}, {bBase}");
            w.L($"s_cselect_b64 {bias}, {w.Pointer("bias2")}, {bias}");
            w.L($"s_cselect_b64 {cBase}, {w.Pointer("y2")}, {cBase}");
            w.L($"s_cselect_b32 {n}, {w.Scalar("n2")}, {n}");
            w.L($"s_mov_b32 {beta}, 0                              ; no beta: y written whole");
            string first = w.S();
            w.L($"s_lshl_b32 {first}, {w.GroupX}, 6");
            w.L($"s_cmp_ge_u32 {first}, {n}                         ; a block past this product's columns: nothing to do");
            w.L($"s_cbranch_scc1 .Ldone");
        }
        else
        {
            (n, beta) = (w.Scalar("n"), w.Scalar("beta"));
            aBase = w.Strided64(w.Pointer("a"), w.GroupZ, w.Pointer("sa"));
            bBase = w.Strided64(w.Pointer("b"), w.GroupZ, w.Pointer("sb"));
            cBase = w.Strided64(w.Pointer("c"), w.GroupZ, w.Pointer("sc"));
        }

        string lane = w.V(), col = w.V(), colc = w.V(), wave = w.S(), t = w.S(), last = w.S(), rg = w.S(), lastRow = w.S();
        w.L($"v_and_b32_e32 {lane}, 63, {w.Tid}");
        w.L($"v_readfirstlane_b32 {t}, {w.Tid}");
        w.L($"s_lshr_b32 {wave}, {t}, 6                         ; the wave: its slice of k");
        w.L($"s_lshl_b32 {t}, {w.GroupX}, 6");
        w.L($"v_add_u32_e32 {col}, {t}, {lane}                ; this lane's column");
        w.L($"s_sub_u32 {last}, {n}, 1");
        w.L($"v_min_u32_e32 {colc}, {last}, {col}             ; read within the matrix");
        if (multi)
        {
            w.L($"s_mov_b32 {rg}, 0                                ; one row group (m ≤ 8)");
        }
        else
        {
            w.L($"s_lshl_b32 {rg}, {w.GroupY}, 3                    ; the block's first row");
        }

        w.L($"s_sub_u32 {lastRow}, {m}, 1");
        string[] rowOffsets = new string[GemvRows];
        for (int i = 0; i < GemvRows; i++)
        {
            rowOffsets[i] = w.S();
            w.L($"s_add_u32 {t}, {rg}, {i}");
            w.L($"s_min_u32 {t}, {t}, {lastRow}");
            w.L($"s_mul_i32 {t}, {t}, {k}");
            w.L($"s_lshl_b32 {rowOffsets[i]}, {t}, 2               ; row {i}'s bytes into a (rows past m read the last)");
        }

        string slice = w.S(), kStart = w.S(), kEnd = w.S();
        w.L($"s_add_u32 {t}, {k}, 15");
        w.L($"s_lshr_b32 {t}, {t}, 4");
        w.L($"s_lshl_b32 {slice}, {t}, 2                          ; k a slice: ⌈k / 16⌉ · 4");
        w.L($"s_mul_i32 {kStart}, {wave}, {slice}");
        w.L($"s_add_u32 {kEnd}, {kStart}, {slice}");
        w.L($"s_min_u32 {kEnd}, {kEnd}, {k}");
        string xPtr = w.Offset64(aBase, kStart);
        string nBytes = w.S(), stepBytes = w.S(), bOffset = w.V();
        w.L($"s_lshl_b32 {nBytes}, {n}, 2");
        w.L($"s_lshl_b32 {stepBytes}, {n}, 4");
        w.L($"s_mul_i32 {t}, {kStart}, {n}");
        w.L($"v_add_u32_e32 {bOffset}, {t}, {colc}");
        w.L($"v_lshlrev_b32_e32 {bOffset}, 2, {bOffset}");
        string[] acc = new string[GemvRows];
        for (int i = 0; i < GemvRows; i++)
        {
            acc[i] = w.V();
            w.L($"v_mov_b32_e32 {acc[i]}, 0");
        }

        string kk = w.S(), end = w.S();
        w.L($"s_mov_b32 {kk}, {kStart}");
        w.Label(".Lmain");
        w.L($"s_add_u32 {end}, {kk}, 4");
        w.L($"s_cmp_gt_u32 {end}, {kEnd}");
        w.L($"s_cbranch_scc1 .Ltail");
        string[] at = [bOffset, w.V(), w.V(), w.V()], weights = [w.V(), w.V(), w.V(), w.V()];
        for (int u = 1; u < 4; u++)
        {
            w.L($"v_add_u32_e32 {at[u]}, {nBytes}, {at[u - 1]}");
        }

        for (int u = 0; u < 4; u++)
        {
            w.L($"global_load_dword {weights[u]}, {at[u]}, {bBase}");
        }

        int[] xs = new int[GemvRows];
        for (int i = 0; i < GemvRows; i++)
        {
            xs[i] = w.SRange(4);
            w.L($"s_load_dwordx4 %s[{xs[i]}:{xs[i] + 3}], {xPtr}, {rowOffsets[i]}");
        }

        for (int u = 0; u < 4; u++)
        {
            for (int i = 0; i < GemvRows; i++)
            {
                w.L($"v_fma_f32 {acc[i]}, %s{xs[i] + u}, {weights[u]}, {acc[i]}");
            }
        }

        w.L($"v_add_u32_e32 {bOffset}, {stepBytes}, {bOffset}");
        w.Add64(xPtr, "16");
        w.L($"s_mov_b32 {kk}, {end}");
        w.L($"s_branch .Lmain");
        w.Label(".Ltail");
        w.L($"s_cmp_ge_u32 {kk}, {kEnd}");
        w.L($"s_cbranch_scc1 .Lsliced");
        string weight = w.V();
        w.L($"global_load_dword {weight}, {bOffset}, {bBase}");
        string[] x1 = new string[GemvRows];
        for (int i = 0; i < GemvRows; i++)
        {
            x1[i] = w.S();
            w.L($"s_load_dword {x1[i]}, {xPtr}, {rowOffsets[i]}");
        }

        for (int i = 0; i < GemvRows; i++)
        {
            w.L($"v_fma_f32 {acc[i]}, {x1[i]}, {weight}, {acc[i]}");
        }

        w.L($"v_add_u32_e32 {bOffset}, {nBytes}, {bOffset}");
        w.Add64(xPtr, "4");
        w.L($"s_add_u32 {kk}, {kk}, 1");
        w.L($"s_branch .Ltail");

        // The slices' sums: each wave's to LDS (area [wave][row][lane]); the first wave adds slices 1, 2, 3 in order.
        w.Label(".Lsliced");
        string laneBytes = w.V(), slot = w.V();
        w.L($"v_lshlrev_b32_e32 {laneBytes}, 2, {lane}");
        w.L($"s_lshl_b32 {t}, {wave}, 11");
        w.L($"v_add_u32_e32 {slot}, {t}, {laneBytes}");
        for (int i = 0; i < GemvRows; i++)
        {
            w.L($"ds_write_b32 {slot}, {acc[i]}{ProductWriter.Offset(256 * i)}");
        }

        w.L($"s_barrier");
        w.L($"s_cmp_lg_u32 {wave}, 0");
        w.L($"s_cbranch_scc1 .Ldone");
        string[,] parts = new string[3, GemvRows];
        for (int p = 0; p < 3; p++)
        {
            for (int i = 0; i < GemvRows; i++)
            {
                parts[p, i] = w.V();
                w.L($"ds_read_b32 {parts[p, i]}, {laneBytes}{ProductWriter.Offset(2048 * (p + 1) + 256 * i)}");
            }
        }

        for (int i = 0; i < GemvRows; i++)
        {
            for (int p = 0; p < 3; p++)
            {
                w.L($"v_add_f32_e32 {acc[i]}, {acc[i]}, {parts[p, i]}");
            }
        }

        if (multi)
        {
            // The bias after the sum, as the CPU adds it to the product's rows.
            string columnBytes = w.V(), value = w.V();
            w.L($"s_cmp_eq_u64 {bias}, 0");
            w.L($"s_cbranch_scc1 .Lbiased");
            w.L($"v_lshlrev_b32_e32 {columnBytes}, 2, {colc}");
            w.L($"global_load_dword {value}, {columnBytes}, {bias}");
            for (int i = 0; i < GemvRows; i++)
            {
                w.L($"v_add_f32_e32 {acc[i]}, {acc[i]}, {value}");
            }

            w.Label(".Lbiased");
        }

        StoreRows(w, acc, rg, m, n, beta, cBase, col, colc, lanes: true);
        w.Label(".Ldone");
        w.L($"s_endpgm");
        return w.Source();
    }

    // Stores row rg + i's value for i < 8 while rg + i < m: c[(rg + i) · n + column] = value (+ beta · c unless beta is
    // ±0), at `column` (`lanes`: each lane its own, those at or past n left out) or at lane 0 alone.
    private static void StoreRows(ProductWriter w, string[] values, string rg, string m, string n, string beta, string cBase, string column, string readColumn,
        bool lanes)
    {
        string magnitude = w.S(), row = w.S(), t = w.S(), mask = w.SPair(), saved = w.SPair();
        w.L($"s_and_b32 {magnitude}, {beta}, 0x7fffffff");
        if (lanes)
        {
            w.L($"v_cmp_gt_u32_e64 {mask}, {n}, {column}");
        }
        else
        {
            w.L($"v_cmp_eq_u32_e64 {mask}, 0, {column}                  ; lane 0 alone");
        }

        for (int i = 0; i < values.Length; i++)
        {
            w.L($"s_add_u32 {row}, {rg}, {i}");
            if (i > 0)
            {
                w.L($"s_cmp_ge_u32 {row}, {m}");
                w.L($"s_cbranch_scc1 .Ldone");
            }

            string offset = w.V();
            w.L($"s_mul_i32 {t}, {row}, {n}");
            w.L($"v_add_u32_e32 {offset}, {t}, {readColumn}");
            w.L($"v_lshlrev_b32_e32 {offset}, 2, {offset}");
            w.L($"s_cmp_eq_u32 {magnitude}, 0");
            w.L($"s_cbranch_scc1 .Lput{i}");
            string old = w.V();
            w.L($"global_load_dword {old}, {offset}, {cBase}");
            w.L($"v_fma_f32 {values[i]}, {beta}, {old}, {values[i]}");
            w.Label($".Lput{i}");
            w.L($"s_and_saveexec_b64 {saved}, {mask}");
            w.L($"global_store_dword {offset}, {values[i]}, {cBase}");
            w.L($"s_or_b64 exec, exec, {saved}");
        }
    }

    // ---- gemv_nt_f32 -----------------------------------------------------------------------------------------------

    private static GcnKernelSource GemvNT()
    {
        var w = new ProductWriter("gemv_nt_f32");
        // Each wave's own LDS: 8 rows of 128 floats (the tree's lanes past the stride read the upper half, never written).
        w.Head(GemvArguments, GemvThreads, lds: 4 * GemvRows * 128 * 4, workgroupIds: 3);
        string m = w.Scalar("m"), n = w.Scalar("n"), k = w.Scalar("k"), beta = w.Scalar("beta");
        string aBase = w.Strided64(w.Pointer("a"), w.GroupZ, w.Pointer("sa"));
        string bBase = w.Strided64(w.Pointer("b"), w.GroupZ, w.Pointer("sb"));
        string cBase = w.Strided64(w.Pointer("c"), w.GroupZ, w.Pointer("sc"));
        string lane = w.V(), wave = w.S(), t = w.S(), j = w.S(), rg = w.S(), lastRow = w.S();
        w.L($"v_and_b32_e32 {lane}, 63, {w.Tid}");
        w.L($"v_readfirstlane_b32 {t}, {w.Tid}");
        w.L($"s_lshr_b32 {wave}, {t}, 6");
        w.L($"s_lshl_b32 {j}, {w.GroupX}, 2");
        w.L($"s_add_u32 {j}, {j}, {wave}                         ; this wave's column");
        w.L($"s_cmp_ge_u32 {j}, {n}");
        w.L($"s_cbranch_scc1 .Ldone");
        w.L($"s_lshl_b32 {rg}, {w.GroupY}, 3");
        w.L($"s_sub_u32 {lastRow}, {m}, 1");
        string[] rows = new string[GemvRows];
        for (int i = 0; i < GemvRows; i++)
        {
            string r = w.S();
            w.L($"s_add_u32 {r}, {rg}, {i}");
            w.L($"s_min_u32 {r}, {r}, {lastRow}");
            rows[i] = w.Strided(aBase, r, k);
        }

        string bRow = w.Strided(bBase, j, k);
        string laneBytes = w.V();
        w.L($"v_lshlrev_b32_e32 {laneBytes}, 2, {lane}");
        string[] acc = new string[GemvRows];
        for (int i = 0; i < GemvRows; i++)
        {
            acc[i] = w.V();
            w.L($"v_mov_b32_e32 {acc[i]}, 0");
        }

        // Four k a lane a step (k = base + lane + 64u), every load first.
        string kb = w.S(), end = w.S(), off = w.V();
        w.L($"s_mov_b32 {kb}, 0");
        w.Label(".Lmain");
        w.L($"s_add_u32 {end}, {kb}, 0x100");
        w.L($"s_cmp_gt_u32 {end}, {k}");
        w.L($"s_cbranch_scc1 .Ltail");
        w.L($"s_lshl_b32 {t}, {kb}, 2");
        w.L($"v_add_u32_e32 {off}, {t}, {laneBytes}");
        string[] weights = new string[4];
        string[,] inputs = new string[GemvRows, 4];
        for (int u = 0; u < 4; u++)
        {
            weights[u] = w.V();
            w.L($"global_load_dword {weights[u]}, {off}, {bRow}{ProductWriter.Offset(256 * u)}");
        }

        for (int i = 0; i < GemvRows; i++)
        {
            for (int u = 0; u < 4; u++)
            {
                inputs[i, u] = w.V();
                w.L($"global_load_dword {inputs[i, u]}, {off}, {rows[i]}{ProductWriter.Offset(256 * u)}");
            }
        }

        for (int u = 0; u < 4; u++)
        {
            for (int i = 0; i < GemvRows; i++)
            {
                w.L($"v_fma_f32 {acc[i]}, {inputs[i, u]}, {weights[u]}, {acc[i]}");
            }
        }

        w.L($"s_mov_b32 {kb}, {end}");
        w.L($"s_branch .Lmain");

        // Then 64 k a step, the lanes past k left out.
        w.Label(".Ltail");
        w.L($"s_cmp_ge_u32 {kb}, {k}");
        w.L($"s_cbranch_scc1 .Lreduce");
        string kIndex = w.V(), saved = w.SPair();
        w.L($"v_add_u32_e32 {kIndex}, {kb}, {lane}");
        w.L($"v_cmp_gt_u32_e32 vcc, {k}, {kIndex}");
        w.L($"s_and_saveexec_b64 {saved}, vcc");
        w.L($"v_lshlrev_b32_e32 {off}, 2, {kIndex}");
        string weight = w.V();
        string[] input = new string[GemvRows];
        w.L($"global_load_dword {weight}, {off}, {bRow}");
        for (int i = 0; i < GemvRows; i++)
        {
            input[i] = w.V();
            w.L($"global_load_dword {input[i]}, {off}, {rows[i]}");
        }

        for (int i = 0; i < GemvRows; i++)
        {
            w.L($"v_fma_f32 {acc[i]}, {input[i]}, {weight}, {acc[i]}");
        }

        w.L($"s_or_b64 exec, exec, {saved}");
        w.L($"s_add_u32 {kb}, {kb}, 64");
        w.L($"s_branch .Ltail");

        // Each row's 64 lane sums by one tree in the wave's LDS (lane l adds lane l + 32, ..., l + 1).
        w.Label(".Lreduce");
        string slot = w.V();
        w.L($"s_lshl_b32 {t}, {wave}, 12");
        w.L($"v_add_u32_e32 {slot}, {t}, {laneBytes}");
        for (int i = 0; i < GemvRows; i++)
        {
            w.L($"ds_write_b32 {slot}, {acc[i]}{ProductWriter.Offset(512 * i)}");
        }

        for (int stride = 32; stride >= 1; stride /= 2)
        {
            string[] other = new string[GemvRows];
            for (int i = 0; i < GemvRows; i++)
            {
                other[i] = w.V();
                w.L($"ds_read_b32 {other[i]}, {slot}{ProductWriter.Offset(512 * i + 4 * stride)}");
            }

            for (int i = 0; i < GemvRows; i++)
            {
                w.L($"v_add_f32_e32 {acc[i]}, {acc[i]}, {other[i]}");
                if (stride > 1)
                {
                    w.L($"ds_write_b32 {slot}, {acc[i]}{ProductWriter.Offset(512 * i)}");
                }
            }
        }

        string column = w.V();
        w.L($"v_mov_b32_e32 {column}, {j}");
        StoreRows(w, acc, rg, m, n, beta, cBase, lane, column, lanes: false);
        w.Label(".Ldone");
        w.L($"s_endpgm");
        return w.Source();
    }

    // ---- The writer ------------------------------------------------------------------------------------------------

    /// <summary>
    /// Writes a product's GCN text with virtual registers: its directives (the block, LDS, the system registers), its
    /// arguments loaded once into one scalar range, 64-bit address arithmetic on the scalar unit, LLVM's unsigned
    /// division by a reciprocal, and the grid-stride loop of the expansions. Each kernel's text is written once a process.
    /// </summary>
    private sealed class ProductWriter(string name)
    {
        private readonly StringBuilder _text = new StringBuilder().Append(CultureInfo.InvariantCulture, $".kernel {name}\n");
        private readonly Dictionary<string, int> _dwords = new(StringComparer.Ordinal);
        private int _s = 100, _v = 100, _labels, _first;

        /// <summary>The thread's index in the block (v0 copied).</summary>
        public string Tid => "%v1";

        /// <summary>The workgroup ids the directives asked for, and the dispatch packet's address (with dispatch).</summary>
        public string GroupX { get; private set; } = "s2";

        public string GroupY { get; private set; } = "s3";

        public string GroupZ { get; private set; } = "s4";

        private bool _dispatch;

        public string S() => F($"%s{_s++}");

        public string SPair()
        {
            int id = (_s + 1) & ~1;
            _s = id + 2;
            return F($"%s[{id}:{id + 1}]");
        }

        /// <summary>The first of <paramref name="count"/> fresh adjacent scalar registers.</summary>
        public int SRange(int count)
        {
            int first = _s;
            _s += count;
            return first;
        }

        public string V() => F($"%v{_v++}");

        /// <summary>The first of <paramref name="count"/> fresh adjacent vector registers.</summary>
        public int Range(int count)
        {
            int first = _v;
            _v += count;
            return first;
        }

        public string NewLabel() => F($".L{_labels++}");

        public void L(FormattableString line) => _text.Append("  ").Append(line.ToString(CultureInfo.InvariantCulture)).Append('\n');

        public void Label(string label) => _text.Append(label).Append(":\n");

        /// <summary>Lines another writer made (each indented already), as they are.</summary>
        public void Lines(StringBuilder lines) => _text.Append(lines);

        /// <summary>Several instruction lines (split at their line breaks), each indented; labels (ending in ':') as they are.</summary>
        public void Block(string lines)
        {
            foreach (var part in lines.Split('\n'))
            {
                string line = part.Trim();
                if (line.Length > 0)
                {
                    _text.Append(line.EndsWith(':') ? "" : "  ").Append(line).Append('\n');
                }
            }
        }

        /// <summary>The kernel arguments' address (s[0:1], or s[2:3] after the dispatch packet's).</summary>
        public string Kernarg => _dispatch ? "s[2:3]" : "s[0:1]";

        private void Raw(FormattableString line) => _text.Append(line.ToString(CultureInfo.InvariantCulture)).Append('\n');

        private static string F(FormattableString text) => text.ToString(CultureInfo.InvariantCulture);

        public static string Offset(int bytes) => bytes == 0 ? "" : F($" offset:{bytes}");

        public GcnKernelSource Source() => new(name, Encoding.UTF8.GetBytes(_text.ToString()));

        public static string Lo(string pair) => F($"%s{Id(pair)}");

        public static string Hi(string pair) => F($"%s{Id(pair) + 1}");

        private static int Id(string pair) => int.Parse(pair.AsSpan(3, pair.IndexOf(':', StringComparison.Ordinal) - 3), CultureInfo.InvariantCulture);

        /// <summary>
        /// Declares the arguments, the block, LDS and the system registers, and loads every argument dword once into one
        /// scalar range (in pieces of 8, 4, 2 and 1 dwords from the start, so a pointer never straddles two).
        /// </summary>
        public void Head(IReadOnlyList<GcnArgument> arguments, int block, int lds = 0, bool dispatch = false, int workgroupIds = 1, int loadDwords = -1)
        {
            int end = 0;
            foreach (var a in arguments)
            {
                Raw($".arg {a.Name}, {a.Size}, {(a.Kind == GcnArgumentKind.GlobalBuffer ? "global" : "value")}");
                _dwords[a.Name!] = a.Offset / 4;
                end = Math.Max(end, a.Offset + a.Size);
            }

            Raw($".reqd_workgroup_size {block}, 1, 1");
            if (lds > 0)
            {
                Raw($".lds {lds}");
            }

            _dispatch = dispatch;
            if (dispatch)
            {
                Raw($".user_sgpr dispatch_ptr");
            }

            Raw($".user_sgpr kernarg_segment_ptr");
            Raw($".system_sgpr {(workgroupIds == 3 ? "workgroup_id_z" : workgroupIds == 2 ? "workgroup_id_y" : "workgroup_id_x")}");
            int user = dispatch ? 4 : 2;
            GroupX = F($"s{user}");
            GroupY = F($"s{user + 1}");
            GroupZ = F($"s{user + 2}");
            string kernarg = dispatch ? "s[2:3]" : "s[0:1]";
            int dwords = loadDwords >= 0 ? loadDwords : (end + 3) / 4;
            _first = SRange(dwords);
            for (int at = 0; at < dwords;)
            {
                int take = dwords - at >= 8 ? 8 : dwords - at >= 4 ? 4 : dwords - at >= 2 ? 2 : 1;
                string target = take == 1 ? F($"%s{_first + at}") : F($"%s[{_first + at}:{_first + at + take - 1}]");
                L($"s_load_dword{(take == 1 ? "" : F($"x{take}"))} {target}, {kernarg}, 0x{4 * at:x}");
                at += take;
            }

            L($"v_mov_b32_e32 {Tid}, v0");
        }

        /// <summary>A pointer argument's register pair (or an 8-byte value's).</summary>
        public string Pointer(string argument)
        {
            int d = _first + _dwords[argument];
            return F($"%s[{d}:{d + 1}]");
        }

        /// <summary>A 4-byte argument's register.</summary>
        public string Scalar(string argument) => F($"%s{_first + _dwords[argument]}");

        /// <summary>pointer + index · stride · 4 (stride a 32-bit element count), in a fresh pair.</summary>
        public string Strided(string pointer, string index, string stride)
        {
            string pair = SPair();
            L($"s_mul_i32 {Lo(pair)}, {index}, {stride}");
            L($"s_mul_hi_u32 {Hi(pair)}, {index}, {stride}");
            L($"s_lshl_b64 {pair}, {pair}, 2");
            L($"s_add_u32 {Lo(pair)}, {Lo(pointer)}, {Lo(pair)}");
            L($"s_addc_u32 {Hi(pair)}, {Hi(pointer)}, {Hi(pair)}");
            return pair;
        }

        /// <summary>pointer + index · stride · 4 (stride a 64-bit element count in a pair), in a fresh pair.</summary>
        public string Strided64(string pointer, string index, string stride)
        {
            string pair = SPair(), t = S();
            L($"s_mul_i32 {Lo(pair)}, {index}, {Lo(stride)}");
            L($"s_mul_hi_u32 {Hi(pair)}, {index}, {Lo(stride)}");
            L($"s_mul_i32 {t}, {index}, {Hi(stride)}");
            L($"s_add_u32 {Hi(pair)}, {Hi(pair)}, {t}");
            L($"s_lshl_b64 {pair}, {pair}, 2");
            L($"s_add_u32 {Lo(pair)}, {Lo(pointer)}, {Lo(pair)}");
            L($"s_addc_u32 {Hi(pair)}, {Hi(pointer)}, {Hi(pair)}");
            return pair;
        }

        /// <summary>pointer + elements · 4 (a 32-bit element count), in a fresh pair.</summary>
        public string Offset64(string pointer, string elements)
        {
            string pair = Bytes64(elements);
            L($"s_add_u32 {Lo(pair)}, {Lo(pointer)}, {Lo(pair)}");
            L($"s_addc_u32 {Hi(pair)}, {Hi(pointer)}, {Hi(pair)}");
            return pair;
        }

        /// <summary>value · 4 as a 64-bit byte count, in a fresh pair.</summary>
        public string Bytes64(string value)
        {
            string pair = SPair();
            L($"s_lshl_b32 {Lo(pair)}, {value}, 2");
            L($"s_lshr_b32 {Hi(pair)}, {value}, 30");
            return pair;
        }

        /// <summary>pair += a small constant or a 32-bit register.</summary>
        public void Add64(string pair, string value)
        {
            L($"s_add_u32 {Lo(pair)}, {Lo(pair)}, {value}");
            L($"s_addc_u32 {Hi(pair)}, {Hi(pair)}, 0");
        }

        /// <summary>pair += another pair.</summary>
        public void Add64Pair(string pair, string value)
        {
            L($"s_add_u32 {Lo(pair)}, {Lo(pair)}, {Lo(value)}");
            L($"s_addc_u32 {Hi(pair)}, {Hi(pair)}, {Hi(value)}");
        }

        /// <summary>LLVM's refined reciprocal of the unsigned scalar <paramref name="d"/> (2^32 / d, to within one), in a vector register.</summary>
        public string Reciprocal(string d)
        {
            string r = V(), t = V(), negative = S();
            L($"v_cvt_f32_u32_e32 {r}, {d}");
            L($"s_sub_i32 {negative}, 0, {d}");
            L($"v_rcp_iflag_f32_e32 {r}, {r}");
            L($"v_mul_f32_e32 {r}, 0x4f7ffffe, {r}");
            L($"v_cvt_u32_f32_e32 {r}, {r}");
            L($"v_mul_lo_u32 {t}, {negative}, {r}");
            L($"v_mul_hi_u32 {t}, {r}, {t}");
            L($"v_add_u32_e32 {r}, {r}, {t}");
            return r;
        }

        /// <summary>x / d and x mod d (unsigned, exact), through <see cref="Reciprocal"/>'s value and two corrections.</summary>
        public (string Quotient, string Remainder) DivMod(string x, string d, string reciprocal)
        {
            string q = V(), r = V(), t = V(), q1 = V();
            L($"v_mul_hi_u32 {q}, {x}, {reciprocal}");
            L($"v_mul_lo_u32 {t}, {q}, {d}");
            L($"v_sub_u32_e32 {r}, {x}, {t}");
            for (int pass = 0; pass < 2; pass++)
            {
                L($"v_add_u32_e32 {q1}, 1, {q}");
                L($"v_subrev_u32_e32 {t}, {d}, {r}");
                L($"v_cmp_le_u32_e32 vcc, {d}, {r}");
                L($"v_cndmask_b32_e32 {q}, {q}, {q1}, vcc");
                L($"v_cndmask_b32_e32 {r}, {r}, {t}, vcc");
            }

            return (q, r);
        }

        // The grid's work-items (the dispatch packet's grid_size_x), read by GridIndex for GridLoop.
        private string _grid = "";

        /// <summary>The work-item's index in the grid (workgroup id x · block + tid), and the grid's size for <see cref="GridLoop"/>.</summary>
        public string GridIndex()
        {
            if (!_dispatch)
            {
                throw new InvalidOperationException("A grid-stride kernel reads the dispatch packet.");
            }

            string index = V(), t = S();
            _grid = S();
            L($"s_load_dword {_grid}, s[0:1], 0xc                    ; the grid's work-items (grid_size_x)");
            L($"s_lshl_b32 {t}, {GroupX}, 8");
            L($"v_add_u32_e32 {index}, {t}, {Tid}");
            return index;
        }

        /// <summary>
        /// Runs <paramref name="body"/> with i = index, index + grid, ... below <paramref name="count"/> (i in a fresh
        /// register, the lanes past it left out), then gives the lanes back: a grid-stride loop that a kernel may run
        /// several times (after <see cref="GridIndex"/>).
        /// </summary>
        public void StrideLoop(string count, string index, Action<string> body)
        {
            string i = V(), saved = SPair(), head = NewLabel(), done = NewLabel();
            L($"v_mov_b32_e32 {i}, {index}");
            L($"v_cmp_gt_u32_e32 vcc, {count}, {i}");
            L($"s_and_saveexec_b64 {saved}, vcc");
            L($"s_cbranch_execz {done}");
            Label(head);
            body(i);
            L($"v_add_u32_e32 {i}, {_grid}, {i}");
            L($"v_cmp_gt_u32_e32 vcc, {count}, {i}");
            L($"s_and_b64 exec, exec, vcc");
            L($"s_cbranch_execnz {head}");
            Label(done);
            L($"s_or_b64 exec, exec, {saved}");
        }

        /// <summary>Runs <paramref name="body"/> for index, index + grid, ... below <paramref name="count"/>, the lanes past it left out; then the kernel's end.</summary>
        public void GridLoop(string count, string index, Action body)
        {
            string saved = SPair();
            L($"v_cmp_gt_u32_e32 vcc, {count}, {index}");
            L($"s_and_saveexec_b64 {saved}, vcc");
            L($"s_cbranch_execz .Lgrid_done");
            Label(".Lgrid");
            body();
            L($"v_add_u32_e32 {index}, {_grid}, {index}");
            L($"v_cmp_gt_u32_e32 vcc, {count}, {index}");
            L($"s_and_b64 exec, exec, vcc");
            L($"s_cbranch_execnz .Lgrid");
            Label(".Lgrid_done");
            L($"s_endpgm");
        }
    }
}
