// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;

namespace Idrak.Gpu.Hip;

// HIP's matrix products in GCN (plan 35, G4a), written for gfx9's wave of 64 as CUDA's are in PTX (Cuda/PtxKernels.cs,
// .Rows.cs, .Quantized.cs): CUDA's names, CUDA's arguments in CUDA's order, the CPU backend's results within the
// device-against-CPU tolerance. Every output's sum is taken in one order fixed by the shapes (rule 86): over k in
// increasing order within a thread, then the parts (slices of k inside a block, splits of k across blocks) added in
// their order, by LDS in a fixed shape or by a second kernel, never by atomics; so a result is the same on every grid,
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
// - {int8,int4,bf16,f16}_gemv[_rR]_f32: the packed few-row products (Backend.Int8MatMul, Int4MatMul, BFloat16MatMul,
//   Float16MatMul for m ≤ R; R = 1, 2, 4, 8, the last without a suffix as CUDA's int8_gemv_f32): one wave of 64 packed
//   words (4, 8, 2, 2 columns each), every weight word read once a split; k in splits of `chunk` rows (grid y, leftover
//   8.10): one split stores y (int8 scaled per column), several store each split's sums in part[split, r, j] and
//   int8_gemv_finish_f32 adds them in split order (and scales them for int8). Four rows of k a step, the four words in
//   flight, each row's four inputs read by the scalar unit. int4: a group's eight scales read once for its 32 rows;
//   `chunk` a multiple of 32 (CUDA's splits start on 64). CUDA's argument list, `counters` unused (no atomics).
// - int8_gemv_finish_f32: y[i] = Σ_split part[split · size + i] in split order, times s[i mod cols] (s = 0: unscaled,
//   for int4, bfloat16 and float16, which CUDA sums in the product kernel).
// - int8_matmul_f32: many rows through int8 weights (CUDA's m > 8 path): a thread a (row, word), k in order.
// - The expansions: int8_dequant_f32, int4_dequant_f32, bf16_dequant_f32, f16_dequant_f32 (a half widened exactly, its
//   NaN payload kept as the CPU keeps it), and bf16_pack_f32 (to nearest, ties to even; NaN to 0x7FC0, the CPU's).
//   A thread a packed word, grid-stride.
//
// Bounds: every matrix below 2^30 elements (byte offsets are 32-bit), grids below 2^29 work-items (grid-stride indexes
// do not wrap); the backend bounds its grids by the device's limits (HipDeviceLimits) and splits larger work.
//
// For --profile labels (CUDA's "{kind}_{nn} {m}x{n}x{k}" with 2·m·n·k operations) and --bench-hip rows: each kernel's
// floating-point operations (a fused multiply-add counts 2) and global-memory bytes, as formulas of its shape; b = batch,
// s = splits, w = words per weight row (⌈n / 4⌉ int8, ⌈n / 8⌉ int4, ⌈n / 2⌉ bfloat16 and float16), g = ⌈k / 32⌉.
//
// | Kernel                 | Arguments                                                        | Launch (block)                  | Flops            | Bytes                                         |
// |------------------------|------------------------------------------------------------------|---------------------------------|------------------|-----------------------------------------------|
// | matmul_f32             | a, b, c, m, n, k, ta, tb, beta, sa, sb, sc, alpha, bias          | ⌈n/64⌉ × ⌈m/64⌉ × b (256)       | 2·m·n·k·b        | 4·b·(m·k + k·n + m·n·(beta ≠ 0 ? 2 : 1)) + 4n |
// | gemv_nn_f32            | a, b, c, m, n, k, beta, sa, sb, sc                               | ⌈n/64⌉ × ⌈m/8⌉ × b (256)        | 2·m·n·k·b        | 4·b·(m·k + ⌈m/8⌉·k·n + m·n)                   |
// | gemv_nt_f32            | a, b, c, m, n, k, beta, sa, sb, sc                               | ⌈n/4⌉ × ⌈m/8⌉ × b (256)         | 2·m·n·k·b        | 4·b·(m·k + ⌈m/8⌉·k·n + m·n)                   |
// | int8_gemv[_rR]_f32     | x, q, s, y, part, m, n, k, n4, chunk, splits, counters           | ⌈w/64⌉ × s (64)                 | 2·m·n·k          | 4·k·w + 4·m·k + 4n + 4·m·n·(s > 1 ? s : 1)    |
// | int4_gemv[_rR]_f32     | (the same)                                                       | ⌈w/64⌉ × s (64)                 | 2·m·n·k + k·n    | 4·k·w + 32·g·w + 4·m·k + 4·m·n·(s > 1 ? s : 1)|
// | bf16_gemv[_rR]_f32     | (the same; s unused)                                             | ⌈w/64⌉ × s (64)                 | 2·m·n·k          | 4·k·w + 4·m·k + 4·m·n·(s > 1 ? s : 1)         |
// | f16_gemv[_rR]_f32      | (the same; s unused)                                             | ⌈w/64⌉ × s (64)                 | 2·m·n·k          | 4·k·w + 4·m·k + 4·m·n·(s > 1 ? s : 1)         |
// | int8_gemv_finish_f32   | part, s, y, cols, size, splits, n                                | ⌈n/256⌉ or fewer (256)          | n·splits         | 4·n·(splits + 1) (+ 4n scaled)                |
// | int8_matmul_f32        | x, q, s, y, k, n4, cols, n (n = m·n4)                            | ⌈m·n4/256⌉ or fewer (256)       | 2·m·n·k          | 4·m·n4·k + 4·m·k·n4 + 4·m·n + 4n              |
// | int8_dequant_f32       | q, s, w, n4, cols, n (n = k·n4)                                  | ⌈n/256⌉ or fewer (256)          | k·n              | 4·k·n4 + 4n + 4·k·n                           |
// | int4_dequant_f32       | q, s, w, words, cols, n (n = k·words)                            | ⌈n/256⌉ or fewer (256)          | k·n              | 4·k·w + 32·g·w + 4·k·n                        |
// | bf16_dequant_f32       | q, w, words, cols, n (n = k·words)                               | ⌈n/256⌉ or fewer (256)          | 0                | 4·k·w + 4·k·n                                 |
// | f16_dequant_f32        | q, w, words, cols, n (n = k·words)                               | ⌈n/256⌉ or fewer (256)          | 0                | 4·k·w + 4·k·n                                 |
// | bf16_pack_f32          | x, y, count, n (n = ⌈count/2⌉)                                   | ⌈n/256⌉ or fewer (256)          | 0                | 4·count + 4n                                  |
//
// (n in a row's Bytes is that kernel's own argument: the output columns for the products, the words for the expansions.)
internal static partial class GcnKernels
{
    /// <summary>The products, the packed few-row products and the expansions, in the order they join <see cref="All"/>.</summary>
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

    /// <summary>Packed words a packed few-row product's block takes (one wave, a word a lane).</summary>
    internal const int PackedGemvWords = 64;

    /// <summary>The block of the expansions, <c>int8_gemv_finish_f32</c> and <c>int8_matmul_f32</c> (grid-stride).</summary>
    internal const int ExpansionBlock = 256;

    /// <summary>Rows of an int4 weight matrix sharing one scale per column (as CUDA's Int4Group).</summary>
    internal const int Int4Group = 32;

    /// <summary>The row counts the packed few-row products are written for; m rows take the first at least m.</summary>
    internal static ReadOnlySpan<int> PackedGemvRowCounts => [1, 2, 4, 8];

    /// <summary>The packed weight formats of the few-row products.</summary>
    internal enum PackedWeights
    {
        Int8,
        Int4,
        BFloat16,
        Float16,
    }

    // Kernel names by [format * 4 + row-count index] (PackedGemvRowCounts), built once: no name is formatted per call.
    private static readonly string[] PackedGemvNames =
    [
        "int8_gemv_r1_f32", "int8_gemv_r2_f32", "int8_gemv_r4_f32", "int8_gemv_f32",
        "int4_gemv_r1_f32", "int4_gemv_r2_f32", "int4_gemv_r4_f32", "int4_gemv_f32",
        "bf16_gemv_r1_f32", "bf16_gemv_r2_f32", "bf16_gemv_r4_f32", "bf16_gemv_f32",
        "f16_gemv_r1_f32", "f16_gemv_r2_f32", "f16_gemv_r4_f32", "f16_gemv_f32",
    ];

    /// <summary>The packed few-row product for <paramref name="m"/> rows (1 to 8) of <paramref name="format"/>'s weights.</summary>
    internal static string PackedGemvKernel(PackedWeights format, int m)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(m, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(m, GemvRows);
        var counts = PackedGemvRowCounts;
        int i = 0;
        while (counts[i] < m)
        {
            i++;
        }

        return PackedGemvNames[(int)format * 4 + i];
    }

    /// <summary>Columns a packed word holds in <paramref name="format"/>.</summary>
    internal static int ColumnsPerWord(PackedWeights format) => format switch
    {
        PackedWeights.Int8 => 4,
        PackedWeights.Int4 => 8,
        _ => 2,
    };

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

    /// <summary>
    /// A packed few-row product's arguments, CUDA's: x, q, s, y, part; m, n, k, n4 (words per weight row), chunk (rows of
    /// k a split), splits; counters (unused: the splits are added by int8_gemv_finish_f32).
    /// </summary>
    internal static IReadOnlyList<GcnArgument> PackedGemvArguments { get; } =
    [
        GcnArgument.Buffer(0, "x"), GcnArgument.Buffer(8, "q"), GcnArgument.Buffer(16, "s"), GcnArgument.Buffer(24, "y"), GcnArgument.Buffer(32, "part"),
        GcnArgument.Value(40, 4, "m"), GcnArgument.Value(44, 4, "n"), GcnArgument.Value(48, 4, "k"), GcnArgument.Value(52, 4, "n4"),
        GcnArgument.Value(56, 4, "chunk"), GcnArgument.Value(60, 4, "splits"), GcnArgument.Buffer(64, "counters"),
    ];

    /// <summary>Bytes of a packed few-row product's arguments.</summary>
    internal const int PackedGemvArgumentBytes = 72;

    /// <summary>int8_gemv_finish_f32's arguments, CUDA's: part, s (0: unscaled), y; cols, size (m·n), splits, n.</summary>
    internal static IReadOnlyList<GcnArgument> GemvFinishArguments { get; } =
    [
        GcnArgument.Buffer(0, "part"), GcnArgument.Buffer(8, "s"), GcnArgument.Buffer(16, "y"), GcnArgument.Value(24, 4, "cols"),
        GcnArgument.Value(28, 4, "size"), GcnArgument.Value(32, 4, "splits"), GcnArgument.Value(36, 4, "n"),
    ];

    /// <summary>Bytes of int8_gemv_finish_f32's arguments.</summary>
    internal const int GemvFinishArgumentBytes = 40;

    /// <summary>int8_matmul_f32's arguments, CUDA's: x, q, s, y; k, n4, cols, n (= m·n4).</summary>
    internal static IReadOnlyList<GcnArgument> Int8MatMulArguments { get; } =
    [
        GcnArgument.Buffer(0, "x"), GcnArgument.Buffer(8, "q"), GcnArgument.Buffer(16, "s"), GcnArgument.Buffer(24, "y"),
        GcnArgument.Value(32, 4, "k"), GcnArgument.Value(36, 4, "n4"), GcnArgument.Value(40, 4, "cols"), GcnArgument.Value(44, 4, "n"),
    ];

    /// <summary>Bytes of int8_matmul_f32's arguments.</summary>
    internal const int Int8MatMulArgumentBytes = 48;

    /// <summary>int8_dequant_f32's and int4_dequant_f32's arguments, CUDA's: q, s, w; words (n4), cols, n (= k·words).</summary>
    internal static IReadOnlyList<GcnArgument> ScaledDequantArguments { get; } =
    [
        GcnArgument.Buffer(0, "q"), GcnArgument.Buffer(8, "s"), GcnArgument.Buffer(16, "w"), GcnArgument.Value(24, 4, "words"),
        GcnArgument.Value(28, 4, "cols"), GcnArgument.Value(32, 4, "n"),
    ];

    /// <summary>Bytes of a scaled expansion's arguments.</summary>
    internal const int ScaledDequantArgumentBytes = 36;

    /// <summary>bf16_dequant_f32's and f16_dequant_f32's arguments, CUDA's: q, w; words, cols, n (= k·words).</summary>
    internal static IReadOnlyList<GcnArgument> HalfDequantArguments { get; } =
    [
        GcnArgument.Buffer(0, "q"), GcnArgument.Buffer(8, "w"), GcnArgument.Value(16, 4, "words"), GcnArgument.Value(20, 4, "cols"),
        GcnArgument.Value(24, 4, "n"),
    ];

    /// <summary>Bytes of a half-width expansion's arguments.</summary>
    internal const int HalfDequantArgumentBytes = 28;

    /// <summary>bf16_pack_f32's arguments, CUDA's: x, y; count (floats), n (words, ⌈count / 2⌉).</summary>
    internal static IReadOnlyList<GcnArgument> PackBFloat16Arguments { get; } =
    [
        GcnArgument.Buffer(0, "x"), GcnArgument.Buffer(8, "y"), GcnArgument.Value(16, 4, "count"), GcnArgument.Value(20, 4, "n"),
    ];

    /// <summary>Bytes of bf16_pack_f32's arguments.</summary>
    internal const int PackBFloat16ArgumentBytes = 24;

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

    /// <summary>
    /// A packed few-row product's arguments (<see cref="PackedGemvArguments"/>) into <paramref name="arguments"/>,
    /// <see cref="PackedGemvArgumentBytes"/> long. <paramref name="scales"/>: int8's per column, int4's per group (0 for the
    /// half-width formats); <paramref name="part"/>: the splits' sums when <paramref name="splits"/> &gt; 1.
    /// </summary>
    internal static void PackPackedGemv(Span<byte> arguments, ulong x, ulong q, ulong scales, ulong y, ulong part, uint m, uint n, uint k, uint words, uint chunk, uint splits)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(arguments.Length, PackedGemvArgumentBytes);
        PackArguments(PackedGemvArguments, arguments, x, q, scales, y, part, m, n, k, words, chunk, splits, 0UL);
    }

    /// <summary>int8_gemv_finish_f32's arguments (<see cref="GemvFinishArguments"/>) into <paramref name="arguments"/>, <see cref="GemvFinishArgumentBytes"/> long.</summary>
    internal static void PackGemvFinish(Span<byte> arguments, ulong part, ulong scales, ulong y, uint cols, uint size, uint splits)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(arguments.Length, GemvFinishArgumentBytes);
        PackArguments(GemvFinishArguments, arguments, part, scales, y, cols, size, splits, size);
    }

    /// <summary>int8_matmul_f32's arguments (<see cref="Int8MatMulArguments"/>) into <paramref name="arguments"/>, <see cref="Int8MatMulArgumentBytes"/> long.</summary>
    internal static void PackInt8MatMul(Span<byte> arguments, ulong x, ulong q, ulong scales, ulong y, uint m, uint n, uint k)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(arguments.Length, Int8MatMulArgumentBytes);
        uint words = (n + 3) / 4;
        PackArguments(Int8MatMulArguments, arguments, x, q, scales, y, k, words, n, m * words);
    }

    /// <summary>
    /// An expansion's arguments into <paramref name="arguments"/>: int8_dequant_f32 and int4_dequant_f32 (<see
    /// cref="ScaledDequantArguments"/>, <see cref="ScaledDequantArgumentBytes"/> long) or bf16_dequant_f32 and
    /// f16_dequant_f32 (<see cref="HalfDequantArguments"/>, <paramref name="scales"/> ignored, <see cref="HalfDequantArgumentBytes"/> long).
    /// </summary>
    internal static void PackDequant(Span<byte> arguments, PackedWeights format, ulong q, ulong scales, ulong w, uint k, uint n)
    {
        uint words = (n + (uint)ColumnsPerWord(format) - 1) / (uint)ColumnsPerWord(format);
        if (format is PackedWeights.Int8 or PackedWeights.Int4)
        {
            ArgumentOutOfRangeException.ThrowIfNotEqual(arguments.Length, ScaledDequantArgumentBytes);
            PackArguments(ScaledDequantArguments, arguments, q, scales, w, words, n, k * words);
            return;
        }

        ArgumentOutOfRangeException.ThrowIfNotEqual(arguments.Length, HalfDequantArgumentBytes);
        PackArguments(HalfDequantArguments, arguments, q, w, words, n, k * words);
    }

    /// <summary>bf16_pack_f32's arguments (<see cref="PackBFloat16Arguments"/>) into <paramref name="arguments"/>, <see cref="PackBFloat16ArgumentBytes"/> long.</summary>
    internal static void PackBFloat16Pack(Span<byte> arguments, ulong x, ulong packed, uint count)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(arguments.Length, PackBFloat16ArgumentBytes);
        PackArguments(PackBFloat16Arguments, arguments, x, packed, count, (count + 1) / 2);
    }

    /// <summary>The expansion kernel of <paramref name="format"/>.</summary>
    internal static string DequantKernel(PackedWeights format) => format switch
    {
        PackedWeights.Int8 => "int8_dequant_f32",
        PackedWeights.Int4 => "int4_dequant_f32",
        PackedWeights.BFloat16 => "bf16_dequant_f32",
        _ => "f16_dequant_f32",
    };

    private static GcnKernelSource[] BuildProducts()
    {
        var kernels = new List<GcnKernelSource> { MatMul(), GemvNN(), GemvNT() };
        foreach (var format in (PackedWeights[])[PackedWeights.Int8, PackedWeights.Int4, PackedWeights.BFloat16, PackedWeights.Float16])
        {
            foreach (int rows in PackedGemvRowCounts)
            {
                kernels.Add(PackedGemv(format, rows));
            }
        }

        kernels.Add(GemvFinish());
        kernels.Add(Int8MatMul());
        kernels.Add(ScaledDequant(int4: false));
        kernels.Add(ScaledDequant(int4: true));
        kernels.Add(HalfDequant(half: false));
        kernels.Add(HalfDequant(half: true));
        kernels.Add(PackBFloat16());
        return [.. kernels];
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

    private static GcnKernelSource GemvNN()
    {
        var w = new ProductWriter("gemv_nn_f32");
        w.Head(GemvArguments, GemvThreads, lds: 4 * GemvRows * GemvNnColumns * 4, workgroupIds: 3);
        string m = w.Scalar("m"), n = w.Scalar("n"), k = w.Scalar("k"), beta = w.Scalar("beta");
        string aBase = w.Strided64(w.Pointer("a"), w.GroupZ, w.Pointer("sa"));
        string bBase = w.Strided64(w.Pointer("b"), w.GroupZ, w.Pointer("sb"));
        string cBase = w.Strided64(w.Pointer("c"), w.GroupZ, w.Pointer("sc"));
        string lane = w.V(), col = w.V(), colc = w.V(), wave = w.S(), t = w.S(), last = w.S(), rg = w.S(), lastRow = w.S();
        w.L($"v_and_b32_e32 {lane}, 63, {w.Tid}");
        w.L($"v_readfirstlane_b32 {t}, {w.Tid}");
        w.L($"s_lshr_b32 {wave}, {t}, 6                         ; the wave: its slice of k");
        w.L($"s_lshl_b32 {t}, {w.GroupX}, 6");
        w.L($"v_add_u32_e32 {col}, {t}, {lane}                ; this lane's column");
        w.L($"s_sub_u32 {last}, {n}, 1");
        w.L($"v_min_u32_e32 {colc}, {last}, {col}             ; read within the matrix");
        w.L($"s_lshl_b32 {rg}, {w.GroupY}, 3                    ; the block's first row");
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

    // ---- The packed few-row products -------------------------------------------------------------------------------

    private static GcnKernelSource PackedGemv(PackedWeights format, int rows)
    {
        int cpw = ColumnsPerWord(format);
        var counts = PackedGemvRowCounts;
        string name = PackedGemvNames[(int)format * 4 + counts.IndexOf(rows)];
        var w = new ProductWriter(name);
        w.Head(PackedGemvArguments, PackedGemvWords, workgroupIds: 2);
        string x = w.Pointer("x"), qp = w.Pointer("q"), sp = w.Pointer("s"), y = w.Pointer("y"), part = w.Pointer("part");
        string m = w.Scalar("m"), n = w.Scalar("n"), k = w.Scalar("k"), words = w.Scalar("n4"), chunk = w.Scalar("chunk"), splits = w.Scalar("splits");
        string word = w.V(), wordc = w.V(), t = w.S(), last = w.S(), kStart = w.S(), kEnd = w.S(), lastRow = w.S();
        w.L($"s_lshl_b32 {t}, {w.GroupX}, 6");
        w.L($"v_add_u32_e32 {word}, {t}, {w.Tid}                 ; this lane's packed word of a row");
        w.L($"s_sub_u32 {last}, {words}, 1");
        w.L($"v_min_u32_e32 {wordc}, {last}, {word}               ; read within the row");
        w.L($"s_mul_i32 {kStart}, {w.GroupY}, {chunk}             ; this split's rows of k");
        w.L($"s_add_u32 {kEnd}, {kStart}, {chunk}");
        w.L($"s_min_u32 {kEnd}, {kEnd}, {k}");
        w.L($"s_sub_u32 {lastRow}, {m}, 1");
        string[] rowOffsets = new string[rows];
        for (int r = 0; r < rows; r++)
        {
            rowOffsets[r] = w.S();
            w.L($"s_min_u32 {t}, {r}, {lastRow}");
            w.L($"s_mul_i32 {t}, {t}, {k}");
            w.L($"s_lshl_b32 {rowOffsets[r]}, {t}, 2               ; row {r}'s bytes into x (rows past m read the last)");
        }

        string xPtr = w.Offset64(x, kStart);
        string rowBytes = w.S(), stepBytes = w.S(), qOffset = w.V();
        w.L($"s_lshl_b32 {rowBytes}, {words}, 2");
        w.L($"s_lshl_b32 {stepBytes}, {words}, 4");
        w.L($"s_mul_i32 {t}, {kStart}, {words}");
        w.L($"v_add_u32_e32 {qOffset}, {t}, {wordc}");
        w.L($"v_lshlrev_b32_e32 {qOffset}, 2, {qOffset}");
        string[,] acc = new string[rows, cpw];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cpw; c++)
            {
                acc[r, c] = w.V();
                w.L($"v_mov_b32_e32 {acc[r, c]}, 0");
            }
        }

        string kk = w.S(), end = w.S(), stop = kEnd;
        int scales = -1;
        w.L($"s_mov_b32 {kk}, {kStart}");
        if (format == PackedWeights.Int4)
        {
            // A group's 32 rows at a time, its eight scales for this lane's columns read once.
            string groupEnd = w.S(), scaleOffset = w.V();
            stop = groupEnd;
            scales = w.Range(8);
            w.Label(".Lgroup");
            w.L($"s_cmp_ge_u32 {kk}, {kEnd}");
            w.L($"s_cbranch_scc1 .Lsummed");
            w.L($"s_and_b32 {t}, {kk}, 0xffffffe0");
            w.L($"s_add_u32 {groupEnd}, {t}, 32");
            w.L($"s_min_u32 {groupEnd}, {groupEnd}, {kEnd}");
            w.L($"s_lshr_b32 {t}, {kk}, 5");
            w.L($"s_mul_i32 {t}, {t}, {words}");
            w.L($"s_lshl_b32 {t}, {t}, 3                             ; the group's row of scales: 8 · words");
            w.L($"v_lshl_add_u32 {scaleOffset}, {wordc}, 3, {t}");
            w.L($"v_lshlrev_b32_e32 {scaleOffset}, 2, {scaleOffset}");
            w.L($"global_load_dwordx4 %v[{scales}:{scales + 3}], {scaleOffset}, {sp}");
            w.L($"global_load_dwordx4 %v[{scales + 4}:{scales + 7}], {scaleOffset}, {sp} offset:16");
        }

        // Four rows of k a step: the four words in flight, each row's four inputs read by the scalar unit.
        w.Label(".Lmain");
        w.L($"s_add_u32 {end}, {kk}, 4");
        w.L($"s_cmp_gt_u32 {end}, {stop}");
        w.L($"s_cbranch_scc1 .Ltail");
        string[] at = [qOffset, w.V(), w.V(), w.V()], packed = [w.V(), w.V(), w.V(), w.V()];
        for (int u = 1; u < 4; u++)
        {
            w.L($"v_add_u32_e32 {at[u]}, {rowBytes}, {at[u - 1]}");
        }

        for (int u = 0; u < 4; u++)
        {
            w.L($"global_load_dword {packed[u]}, {at[u]}, {qp}");
        }

        int[] xs = new int[rows];
        for (int r = 0; r < rows; r++)
        {
            xs[r] = w.SRange(4);
            w.L($"s_load_dwordx4 %s[{xs[r]}:{xs[r] + 3}], {xPtr}, {rowOffsets[r]}");
        }

        for (int u = 0; u < 4; u++)
        {
            string[] decoded = Decode(w, format, packed[u], scales);
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cpw; c++)
                {
                    w.L($"v_fma_f32 {acc[r, c]}, %s{xs[r] + u}, {decoded[c]}, {acc[r, c]}");
                }
            }
        }

        w.L($"v_add_u32_e32 {qOffset}, {stepBytes}, {qOffset}");
        w.Add64(xPtr, "16");
        w.L($"s_mov_b32 {kk}, {end}");
        w.L($"s_branch .Lmain");

        // Then a row at a time.
        w.Label(".Ltail");
        w.L($"s_cmp_ge_u32 {kk}, {stop}");
        w.L($"s_cbranch_scc1 {(format == PackedWeights.Int4 ? ".Lgroup" : ".Lsummed")}");
        string single = w.V();
        w.L($"global_load_dword {single}, {qOffset}, {qp}");
        string[] x1 = new string[rows];
        for (int r = 0; r < rows; r++)
        {
            x1[r] = w.S();
            w.L($"s_load_dword {x1[r]}, {xPtr}, {rowOffsets[r]}");
        }

        string[] one = Decode(w, format, single, scales);
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cpw; c++)
            {
                w.L($"v_fma_f32 {acc[r, c]}, {x1[r]}, {one[c]}, {acc[r, c]}");
            }
        }

        w.L($"v_add_u32_e32 {qOffset}, {rowBytes}, {qOffset}");
        w.Add64(xPtr, "4");
        w.L($"s_add_u32 {kk}, {kk}, 1");
        w.L($"s_branch .Ltail");

        // One split: y (int8 scaled per column); several: this split's sums at part[(split · m + r) · n + column].
        w.Label(".Lsummed");
        string firstColumn = w.V(), saved = w.SPair();
        w.L($"v_lshlrev_b32_e32 {firstColumn}, {int.Log2(cpw)}, {word}");
        string[] columns = new string[cpw], masks = new string[cpw];
        for (int c = 0; c < cpw; c++)
        {
            columns[c] = w.V();
            masks[c] = w.SPair();
            w.L($"v_add_u32_e32 {columns[c]}, {c}, {firstColumn}");
            w.L($"v_cmp_gt_u32_e64 {masks[c]}, {n}, {columns[c]}");
        }

        w.L($"s_cmp_eq_u32 {splits}, 1");
        w.L($"s_cbranch_scc0 .Lpartial");
        string[] scaleOf = new string[cpw];
        if (format == PackedWeights.Int8)
        {
            for (int c = 0; c < cpw; c++)
            {
                string address = w.V();
                scaleOf[c] = w.V();
                w.L($"v_lshlrev_b32_e32 {address}, 2, {columns[c]}");
                w.L($"v_cndmask_b32_e64 {address}, 0, {address}, {masks[c]}");
                w.L($"global_load_dword {scaleOf[c]}, {address}, {sp}");
            }
        }

        StorePacked(w, acc, rows, cpw, m, n, columns, masks, saved, y, rowStart: null, scaleOf: format == PackedWeights.Int8 ? scaleOf : null, label: "y");
        w.L($"s_branch .Ldone");
        w.Label(".Lpartial");
        string first = w.S();
        w.L($"s_mul_i32 {first}, {w.GroupY}, {m}                 ; this split's first row of part");
        StorePacked(w, acc, rows, cpw, m, n, columns, masks, saved, part, rowStart: first, scaleOf: null, label: "part");
        w.Label(".Ldone");
        w.L($"s_endpgm");
        return w.Source();
    }

    // Stores the rows below m of `acc` (times the column's scale when given) at target[(rowStart + r) · n + column], the
    // columns at or past n left out.
    private static void StorePacked(ProductWriter w, string[,] acc, int rows, int cpw, string m, string n, string[] columns, string[] masks, string saved,
        string target, string? rowStart, string[]? scaleOf, string label)
    {
        string row = w.S(), t = w.S();
        for (int r = 0; r < rows; r++)
        {
            if (r > 0)
            {
                w.L($"s_cmp_le_u32 {m}, {r}");
                w.L($"s_cbranch_scc1 .L{label}_stored");
            }

            if (rowStart is null)
            {
                w.L($"s_mul_i32 {t}, {r}, {n}");
            }
            else
            {
                w.L($"s_add_u32 {row}, {rowStart}, {r}");
                w.L($"s_mul_i32 {t}, {row}, {n}");
            }

            for (int c = 0; c < cpw; c++)
            {
                string offset = w.V(), value = acc[r, c];
                w.L($"v_add_u32_e32 {offset}, {t}, {columns[c]}");
                w.L($"v_lshlrev_b32_e32 {offset}, 2, {offset}");
                if (scaleOf is not null)
                {
                    value = w.V();
                    w.L($"v_mul_f32_e32 {value}, {acc[r, c]}, {scaleOf[c]}");
                }

                w.L($"s_and_saveexec_b64 {saved}, {masks[c]}");
                w.L($"global_store_dword {offset}, {value}, {target}");
                w.L($"s_or_b64 exec, exec, {saved}");
            }
        }

        w.Label($".L{label}_stored");
    }

    // A packed word's columns as floats, in fresh registers (int4: times the group's scales, from `scales` on).
    private static string[] Decode(ProductWriter w, PackedWeights format, string packed, int scales)
    {
        int cpw = ColumnsPerWord(format);
        string[] values = new string[cpw];
        for (int c = 0; c < cpw; c++)
        {
            values[c] = w.V();
        }

        switch (format)
        {
            case PackedWeights.Int8:
            case PackedWeights.Int4:
                int bits = format == PackedWeights.Int8 ? 8 : 4;
                for (int c = 0; c < cpw; c++)
                {
                    w.L($"v_bfe_i32 {values[c]}, {packed}, {bits * c}, {bits}");
                    w.L($"v_cvt_f32_i32_e32 {values[c]}, {values[c]}");
                    if (format == PackedWeights.Int4)
                    {
                        w.L($"v_mul_f32_e32 {values[c]}, {values[c]}, %v{scales + c}");
                    }
                }

                break;
            case PackedWeights.BFloat16:
                w.L($"v_lshlrev_b32_e32 {values[0]}, 16, {packed}          ; bfloat16 is a float32's high half");
                w.L($"v_and_b32_e32 {values[1]}, 0xffff0000, {packed}");
                break;
            default:
                string high = w.V();
                w.L($"v_cvt_f32_f16_e32 {values[0]}, {packed}");
                w.L($"v_lshrrev_b32_e32 {high}, 16, {packed}");
                w.L($"v_cvt_f32_f16_e32 {values[1]}, {high}");
                break;
        }

        return values;
    }

    // ---- int8_gemv_finish_f32 --------------------------------------------------------------------------------------

    private static GcnKernelSource GemvFinish()
    {
        var w = new ProductWriter("int8_gemv_finish_f32");
        w.Head(GemvFinishArguments, ExpansionBlock, dispatch: true);
        string part = w.Pointer("part"), sp = w.Pointer("s"), y = w.Pointer("y");
        string cols = w.Scalar("cols"), size = w.Scalar("size"), splits = w.Scalar("splits"), count = w.Scalar("n");
        string index = w.GridIndex();
        string reciprocal = w.Reciprocal(cols);
        string splitBytes = w.Bytes64(size);
        w.GridLoop(count, index, () =>
        {
            string offset = w.V(), sum = w.V(), value = w.V(), at = w.SPair(), split = w.S();
            w.L($"v_lshlrev_b32_e32 {offset}, 2, {index}");
            w.L($"v_mov_b32_e32 {sum}, 0");
            w.L($"s_mov_b64 {at}, {part}");
            w.L($"s_mov_b32 {split}, 0");
            w.Label(".Lsplits");
            w.L($"s_cmp_ge_u32 {split}, {splits}");
            w.L($"s_cbranch_scc1 .Lsummed");
            w.L($"global_load_dword {value}, {offset}, {at}");
            w.L($"v_add_f32_e32 {sum}, {sum}, {value}");
            w.Add64Pair(at, splitBytes);
            w.L($"s_add_u32 {split}, {split}, 1");
            w.L($"s_branch .Lsplits");
            w.Label(".Lsummed");
            w.L($"s_cmp_eq_u64 {sp}, 0");
            w.L($"s_cbranch_scc1 .Lstore");
            var (_, column) = w.DivMod(index, cols, reciprocal);
            string scale = w.V();
            w.L($"v_lshlrev_b32_e32 {column}, 2, {column}");
            w.L($"global_load_dword {scale}, {column}, {sp}");
            w.L($"v_mul_f32_e32 {sum}, {sum}, {scale}");
            w.Label(".Lstore");
            w.L($"global_store_dword {offset}, {sum}, {y}");
        });
        return w.Source();
    }

    // ---- int8_matmul_f32 -------------------------------------------------------------------------------------------

    private static GcnKernelSource Int8MatMul()
    {
        var w = new ProductWriter("int8_matmul_f32");
        w.Head(Int8MatMulArguments, ExpansionBlock, dispatch: true);
        string x = w.Pointer("x"), qp = w.Pointer("q"), sp = w.Pointer("s"), y = w.Pointer("y");
        string k = w.Scalar("k"), words = w.Scalar("n4"), cols = w.Scalar("cols"), count = w.Scalar("n");
        string index = w.GridIndex();
        string reciprocal = w.Reciprocal(words);
        string rowBytes = w.S(), stepBytes = w.S();
        w.L($"s_lshl_b32 {rowBytes}, {words}, 2");
        w.L($"s_lshl_b32 {stepBytes}, {words}, 4");
        w.GridLoop(count, index, () =>
        {
            var (row, wc) = w.DivMod(index, words, reciprocal);
            string xOffset = w.V(), qOffset = w.V(), kk = w.S(), end = w.S();
            w.L($"v_mul_lo_u32 {xOffset}, {row}, {k}");
            w.L($"v_lshlrev_b32_e32 {xOffset}, 2, {xOffset}");
            w.L($"v_lshlrev_b32_e32 {qOffset}, 2, {wc}");
            string[] acc = [w.V(), w.V(), w.V(), w.V()];
            foreach (string a in acc)
            {
                w.L($"v_mov_b32_e32 {a}, 0");
            }

            w.L($"s_mov_b32 {kk}, 0");
            w.Label(".Lmain");
            w.L($"s_add_u32 {end}, {kk}, 4");
            w.L($"s_cmp_gt_u32 {end}, {k}");
            w.L($"s_cbranch_scc1 .Ltail");
            string[] at = [qOffset, w.V(), w.V(), w.V()], packed = [w.V(), w.V(), w.V(), w.V()], inputs = [w.V(), w.V(), w.V(), w.V()];
            for (int u = 1; u < 4; u++)
            {
                w.L($"v_add_u32_e32 {at[u]}, {rowBytes}, {at[u - 1]}");
            }

            for (int u = 0; u < 4; u++)
            {
                w.L($"global_load_dword {packed[u]}, {at[u]}, {qp}");
                w.L($"global_load_dword {inputs[u]}, {xOffset}, {x}{ProductWriter.Offset(4 * u)}");
            }

            for (int u = 0; u < 4; u++)
            {
                string[] decoded = Decode(w, PackedWeights.Int8, packed[u], -1);
                for (int c = 0; c < 4; c++)
                {
                    w.L($"v_fma_f32 {acc[c]}, {inputs[u]}, {decoded[c]}, {acc[c]}");
                }
            }

            w.L($"v_add_u32_e32 {qOffset}, {stepBytes}, {qOffset}");
            w.L($"v_add_u32_e32 {xOffset}, 16, {xOffset}");
            w.L($"s_mov_b32 {kk}, {end}");
            w.L($"s_branch .Lmain");
            w.Label(".Ltail");
            w.L($"s_cmp_ge_u32 {kk}, {k}");
            w.L($"s_cbranch_scc1 .Lsummed");
            string single = w.V(), input = w.V();
            w.L($"global_load_dword {single}, {qOffset}, {qp}");
            w.L($"global_load_dword {input}, {xOffset}, {x}");
            string[] one = Decode(w, PackedWeights.Int8, single, -1);
            for (int c = 0; c < 4; c++)
            {
                w.L($"v_fma_f32 {acc[c]}, {input}, {one[c]}, {acc[c]}");
            }

            w.L($"v_add_u32_e32 {qOffset}, {rowBytes}, {qOffset}");
            w.L($"v_add_u32_e32 {xOffset}, 4, {xOffset}");
            w.L($"s_add_u32 {kk}, {kk}, 1");
            w.L($"s_branch .Ltail");
            w.Label(".Lsummed");
            string firstColumn = w.V(), rowStart = w.V(), saved = w.SPair();
            w.L($"v_lshlrev_b32_e32 {firstColumn}, 2, {wc}");
            w.L($"v_mul_lo_u32 {rowStart}, {row}, {cols}");
            for (int c = 0; c < 4; c++)
            {
                string column = w.V(), mask = w.SPair(), scale = w.V(), offset = w.V(), value = w.V();
                w.L($"v_add_u32_e32 {column}, {c}, {firstColumn}");
                w.L($"v_cmp_gt_u32_e64 {mask}, {cols}, {column}");
                w.L($"s_and_saveexec_b64 {saved}, {mask}");
                w.L($"v_add_u32_e32 {offset}, {rowStart}, {column}");
                w.L($"v_lshlrev_b32_e32 {offset}, 2, {offset}");
                w.L($"v_lshlrev_b32_e32 {column}, 2, {column}");
                w.L($"global_load_dword {scale}, {column}, {sp}");
                w.L($"v_mul_f32_e32 {value}, {acc[c]}, {scale}");
                w.L($"global_store_dword {offset}, {value}, {y}");
                w.L($"s_or_b64 exec, exec, {saved}");
            }
        });
        return w.Source();
    }

    // ---- The expansions --------------------------------------------------------------------------------------------

    // w[row, col] = q's value · s (int8: s[col]; int4: s[row / 32, col], rows of 8 · words scales), a word a thread.
    private static GcnKernelSource ScaledDequant(bool int4)
    {
        var w = new ProductWriter(int4 ? "int4_dequant_f32" : "int8_dequant_f32");
        w.Head(ScaledDequantArguments, ExpansionBlock, dispatch: true);
        string qp = w.Pointer("q"), sp = w.Pointer("s"), wp = w.Pointer("w");
        string words = w.Scalar("words"), cols = w.Scalar("cols"), count = w.Scalar("n");
        int cpw = int4 ? 8 : 4;
        string index = w.GridIndex();
        string reciprocal = w.Reciprocal(words);
        w.GridLoop(count, index, () =>
        {
            var (row, wc) = w.DivMod(index, words, reciprocal);
            string offset = w.V(), packed = w.V(), firstColumn = w.V(), rowStart = w.V(), scaleRow = w.V(), saved = w.SPair();
            w.L($"v_lshlrev_b32_e32 {offset}, 2, {index}");
            w.L($"global_load_dword {packed}, {offset}, {qp}");
            w.L($"v_lshlrev_b32_e32 {firstColumn}, {int.Log2(cpw)}, {wc}");
            w.L($"v_mul_lo_u32 {rowStart}, {row}, {cols}");
            if (int4)
            {
                w.L($"v_lshrrev_b32_e32 {scaleRow}, 5, {row}");
                w.L($"v_mul_lo_u32 {scaleRow}, {scaleRow}, {words}");
                w.L($"v_lshlrev_b32_e32 {scaleRow}, 3, {scaleRow}             ; the group's row of 8 · words scales");
            }

            for (int c = 0; c < cpw; c++)
            {
                string column = w.V(), mask = w.SPair(), value = w.V(), scale = w.V(), at = w.V(), target = w.V();
                w.L($"v_add_u32_e32 {column}, {c}, {firstColumn}");
                w.L($"v_cmp_gt_u32_e64 {mask}, {cols}, {column}");
                w.L($"s_and_saveexec_b64 {saved}, {mask}");
                w.L(int4 ? (FormattableString)$"v_add_u32_e32 {at}, {scaleRow}, {column}" : $"v_mov_b32_e32 {at}, {column}");
                w.L($"v_lshlrev_b32_e32 {at}, 2, {at}");
                w.L($"global_load_dword {scale}, {at}, {sp}");
                w.L($"v_bfe_i32 {value}, {packed}, {(int4 ? 4 : 8) * c}, {(int4 ? 4 : 8)}");
                w.L($"v_cvt_f32_i32_e32 {value}, {value}");
                w.L($"v_mul_f32_e32 {value}, {value}, {scale}");
                w.L($"v_add_u32_e32 {target}, {rowStart}, {column}");
                w.L($"v_lshlrev_b32_e32 {target}, 2, {target}");
                w.L($"global_store_dword {target}, {value}, {wp}");
                w.L($"s_or_b64 exec, exec, {saved}");
            }
        });
        return w.Source();
    }

    // w[row, 2c], w[row, 2c + 1] = the bfloat16 or IEEE half pair of word (row, c), a word a thread; a half widened
    // exactly (subnormals kept), a NaN half to the float NaN with its payload (sign | 0x7F800000 | mantissa << 13), as
    // the CPU's HalfFloats widens it.
    private static GcnKernelSource HalfDequant(bool half)
    {
        var w = new ProductWriter(half ? "f16_dequant_f32" : "bf16_dequant_f32");
        w.Head(HalfDequantArguments, ExpansionBlock, dispatch: true);
        string qp = w.Pointer("q"), wp = w.Pointer("w");
        string words = w.Scalar("words"), cols = w.Scalar("cols"), count = w.Scalar("n");
        string index = w.GridIndex();
        string reciprocal = w.Reciprocal(words);
        string exponent = w.S();
        w.L($"s_mov_b32 {exponent}, 0x7f800000");
        w.GridLoop(count, index, () =>
        {
            var (row, wc) = w.DivMod(index, words, reciprocal);
            string offset = w.V(), packed = w.V(), column = w.V(), target = w.V(), low = w.V(), high = w.V(), mask = w.SPair(), saved = w.SPair();
            w.L($"v_lshlrev_b32_e32 {offset}, 2, {index}");
            w.L($"global_load_dword {packed}, {offset}, {qp}");
            w.L($"v_lshlrev_b32_e32 {column}, 1, {wc}");
            w.L($"v_mul_lo_u32 {target}, {row}, {cols}");
            w.L($"v_add_u32_e32 {target}, {target}, {column}");
            w.L($"v_lshlrev_b32_e32 {target}, 2, {target}");
            if (half)
            {
                string upper = w.V();
                Widen(w, low, packed, exponent);
                w.L($"v_lshrrev_b32_e32 {upper}, 16, {packed}");
                Widen(w, high, upper, exponent);
            }
            else
            {
                w.L($"v_lshlrev_b32_e32 {low}, 16, {packed}");
                w.L($"v_and_b32_e32 {high}, 0xffff0000, {packed}");
            }

            w.L($"global_store_dword {target}, {low}, {wp}");
            w.L($"v_add_u32_e32 {column}, 1, {column}");
            w.L($"v_cmp_gt_u32_e64 {mask}, {cols}, {column}");
            w.L($"s_and_saveexec_b64 {saved}, {mask}");
            w.L($"global_store_dword {target}, {high}, {wp} offset:4");
            w.L($"s_or_b64 exec, exec, {saved}");
        });
        return w.Source();
    }

    // target = the float32 of the half in source's low 16 bits, a NaN's payload kept.
    private static void Widen(ProductWriter w, string target, string source, string exponent)
    {
        string magnitude = w.V(), sign = w.V(), mantissa = w.V(), nan = w.V();
        w.L($"v_cvt_f32_f16_e32 {target}, {source}");
        w.L($"v_and_b32_e32 {magnitude}, 0x7fff, {source}");
        w.L($"v_cmp_lt_u32_e32 vcc, 0x7c00, {magnitude}             ; a NaN half");
        w.L($"v_lshlrev_b32_e32 {sign}, 16, {source}");
        w.L($"v_and_b32_e32 {sign}, 0x80000000, {sign}");
        w.L($"v_and_b32_e32 {mantissa}, 0x3ff, {source}");
        w.L($"v_lshlrev_b32_e32 {mantissa}, 13, {mantissa}");
        w.L($"v_or3_b32 {nan}, {sign}, {mantissa}, {exponent}");
        w.L($"v_cndmask_b32_e32 {target}, {target}, {nan}, vcc");
    }

    // packed[i] = bf16(x[2i]) | bf16(x[2i + 1]) << 16 (0 past count): to nearest, ties to even; NaN to 0x7FC0.
    private static GcnKernelSource PackBFloat16()
    {
        var w = new ProductWriter("bf16_pack_f32");
        w.Head(PackBFloat16Arguments, ExpansionBlock, dispatch: true);
        string x = w.Pointer("x"), y = w.Pointer("y"), count = w.Scalar("count"), words = w.Scalar("n");
        string index = w.GridIndex();
        string bias = w.S(), quiet = w.V();
        w.L($"s_movk_i32 {bias}, 0x7fff");
        w.L($"v_mov_b32_e32 {quiet}, 0x7fc0");
        w.GridLoop(words, index, () =>
        {
            string offset = w.V(), second = w.V(), mask = w.SPair(), address = w.V(), first = w.V(), next = w.V(), low = w.V(), high = w.V(), output = w.V();
            w.L($"v_lshlrev_b32_e32 {offset}, 3, {index}");
            w.L($"global_load_dword {first}, {offset}, {x}");
            w.L($"v_lshl_add_u32 {second}, {index}, 1, 1");
            w.L($"v_cmp_gt_u32_e64 {mask}, {count}, {second}");
            w.L($"v_add_u32_e32 {address}, 4, {offset}");
            w.L($"v_cndmask_b32_e64 {address}, 0, {address}, {mask}");
            w.L($"global_load_dword {next}, {address}, {x}");
            Round(w, low, first, bias, quiet);
            Round(w, high, next, bias, quiet);
            w.L($"v_cndmask_b32_e64 {high}, 0, {high}, {mask}");
            w.L($"v_lshl_or_b32 {output}, {high}, 16, {low}");
            w.L($"v_lshlrev_b32_e32 {offset}, 2, {index}");
            w.L($"global_store_dword {offset}, {output}, {y}");
        });
        return w.Source();
    }

    // target = value's bfloat16 bits: (u + 0x7FFF + ((u >> 16) & 1)) >> 16, or 0x7FC0 for a NaN.
    private static void Round(ProductWriter w, string target, string value, string bias, string quiet)
    {
        string odd = w.V();
        w.L($"v_bfe_u32 {odd}, {value}, 16, 1");
        w.L($"v_add3_u32 {target}, {value}, {bias}, {odd}");
        w.L($"v_lshrrev_b32_e32 {target}, 16, {target}");
        w.L($"v_cmp_u_f32_e32 vcc, {value}, {value}");
        w.L($"v_cndmask_b32_e32 {target}, {target}, {quiet}, vcc");
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
        public void Head(IReadOnlyList<GcnArgument> arguments, int block, int lds = 0, bool dispatch = false, int workgroupIds = 1)
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
            int dwords = (end + 3) / 4;
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
