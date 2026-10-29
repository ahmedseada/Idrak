// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;

namespace Idrak.Gpu.Hip;

// HIP's decoding kernels in GCN (plan 35, G5a), with CUDA's names and arguments where CUDA has the kernel
// (Cuda/PtxKernels.Decoding.cs, .Decoder.cs, .Quantized.cs, .Advanced.cs), computing what the CPU backend computes
// (CpuBackend.Decoding.cs, .Quantized.cs, .Advanced.cs, CpuMath.cs) within the device-against-CPU tolerance.
//
// - The fused packed few-row products (Backend.FusedKernels, rule 82: fewer launches a token), written by the packed
//   products' generator (GcnKernels.Products.cs, PackedGemv) for 1, 2, 4 and 8 rows like the plain ones:
//   {int8,int4,bf16}_gemv_{silu,gelu}[_rR]_f32, the gated products, read their input row as act(gate) · up, made as it
//   is read (CpuMath.Gated's formulas: SiLU as g / (1 + e^-g), GELU's tanh form): a lane makes one of the step's 4 · R
//   values, v_readlane hands each to the scalar unit, which then feeds the product as the plain kernels' scalar loads do.
//   {int8,int4,bf16}_gemv_multi[_rR]_f32, the multi-output products, take products of one input in grid z (each its
//   own weights, scales, output, bias and width), so the queries, keys and values (or gate and up) are one launch. The
//   k splits are the plain products' (HipBackend.GemvSplitCount); their sums are added in split order by the last block
//   of each column range, which a completion counter names (GcnKernels.Finish.cs; rule 86 forbids adding with atomics,
//   not signalling), as CUDA's kernels do: one launch. For a gate/up pair (act not 0) the second of the two products to
//   finish a column range writes act(gate) · up for it. {int8,int4,bf16}_gemv_addnorm[_rR]_f32 (CUDA's
//   *_gemv_addnorm_f32 and *_gemv_normaddnorm_f32, the sandwich block's y norm by flags bit 0) are the plain product
//   whose last block of all then adds the residual and takes the norms, gemv_addnorm_f32's passes on one wave (the same
//   bits). With counters 0 the kernels leave the splits (and the pair) to a second launch, as before G6b:
//   int8_gemv_finish_f32, gemv_multi_finish_f32, and gemv_addnorm_f32 after the plain product (IDRAK_HIP_FINISH=launch).
// - attention_rows_f32 and attention_rows_bf16: decoding over a float32 or bfloat16 cache, each row from its sequence's
//   start (GcnKernels.Attention.cs's kernel with a table of starts); segment_spans_f32 writes the key range of each row
//   of packed sequences (causal, from the sequence's start, a window), so attention_spans_f32 runs them.
// - The rows of decoding: argmax_f32, class_match_f32, add_rms_norm_affine_f32, rms_norm_rope_f32 (a block a row, the
//   reductions in RowWriter's fixed tree), rope_f32, one_hot_f32 and history_push_f32 (an element a work-item,
//   grid-stride), penalize_rows_f32 (a block a row: the row copied, then each distinct id of the window penalized once).
// - The norm and rotation of the attention layer's heads in one launch (rule 82): rms_norm_rope2_f32 (queries and keys,
//   Backend.RmsNormRopePair) and norm_rope_heads_f32 (queries, keys and values into the layouts attention reads, the
//   keys and values into a float32 or bfloat16 cache, Backend.NormRopeHeads), written by rms_norm_rope_f32's generator.
// - The sampler (Backend.SampleRows): sample_rows_f32 over the whole row, or for top-k ≤ 64 over a large vocabulary
//   topk_candidates_f32 then sample_candidates_f32 (CUDA's two stages), and for greedy rows topk_greedy_f32 alone: the
//   last of a row's blocks (a completion counter) samples the row from its blocks' candidates and parts as
//   sample_candidates_greedy_f32 does (a row with one best score finished from the parts), one launch a step (with
//   counters 0, sample_candidates_greedy_f32 in a second launch). Each draw is the CPU's: its counter-based random value
//   bit for bit, its cut-offs the CPU's (ln(min-p) computed on the host), its weights LLVM's exp (see Sampler's
//   remarks). CUDA's atomic flags become a block's own flag and parts, folded in a fixed order (rule 86).
//
// For the profile's labels and --bench-hip's rates (HipBackend.Cost, HipBackend.ProductCost): each kernel, its
// arguments (pointers, then 32-bit values, float ones as their bits; PackArguments packs them), its launch, and its
// floating-point operations and global-memory bytes per unit (a division, exp or rsq counted as the instructions do):
//
// | Kernel                    | Arguments                                                                               | Launch (block)                   | Unit            | Flops | Bytes |
// |---------------------------|-----------------------------------------------------------------------------------------|----------------------------------|-----------------|-------|-------|
// | *_gemv_silu/gelu[_rR]_f32 | the packed product's (x is the gate), then up                                           | ⌈w/64⌉ × splits (64)             | as *_gemv_f32 (ProductCost)     |
// | *_gemv_multi[_rR]_f32     | x, part, m, k, chunk, splits, counters, blocks, nmax; 3 × (q, s, y, bias, n); act, hidden | ⌈w/64⌉ × splits × products (64) | as *_gemv_f32 (ProductCost)     |
// | *_gemv_addnorm[_rR]_f32   | the packed product's, then residual, sum, gain, normalized, ygain; eps, offset, yeps, yoffset, flags | ⌈w/64⌉ × splits (64) | output | as *_gemv_f32 + 9 | + 36 |
// | gemv_multi_finish_f32     | part, hidden, splits, m, nmax, act; 3 × (s, y, bias, n)                                 | ⌈m·n/256⌉ or fewer × products     | output          | 1 + splits | 4·(splits + 2) |
// | gemv_addnorm_f32          | part, s, y, residual, sum, gain, normalized, ygain; cols, splits, size, eps, offset, yeps, yoffset, flags, n | rows (256) | element | 8 + splits | 4·(splits + 8) |
// | attention_rows_f32        | attention_decode_f32's (counters after part), then starts; hpr before n                 | rows x splits (256)              | row × position × dim | 4 | 8 |
// | attention_rows_bf16       | attention_decode_bf16's (counters after part), then starts; hpr before n                | rows x splits (256)              | row × position × dim | 4 | 4 |
// | segment_spans_f32         | starts, begins, ends; rows, steps, window, n                                            | ⌈n/256⌉ or fewer (256)           | row             | 0     | 12    |
// | argmax_f32                | x, y; cols, n                                                                           | rows (256)                       | element         | 1     | 4     |
// | class_match_f32           | predictions, targets, y; cols, threshold, n                                             | rows (256)                       | element         | 2     | 8     |
// | add_rms_norm_affine_f32   | a, b, sum, gain, y; cols, eps, offset, n                                                | rows (256)                       | element         | 6     | 24    |
// | rms_norm_rope_f32         | x, gain, cos, sin, positions, y; cols, eps, offset, heads, steps, half, interleaved, n   | rows (256)                       | element         | 7     | 20    |
// | rope_f32                  | x, y, cos, sin, positions; heads, steps, dim, half, interleaved, sign, n                | ⌈n/256⌉ or fewer (256)           | pair            | 7     | 24    |
// | one_hot_f32               | indices, y; classes, last, n                                                            | ⌈n/256⌉ or fewer (256)           | element         | 0     | 8     |
// | history_push_f32          | ids, history, length; capacity, rows, n                                                 | ⌈n/256⌉ or fewer (256)           | row             | 0     | 8     |
// | penalize_rows_f32         | logits, work, history, length; vocabulary, stride, offset, capacity, lastn, repeat, presence, frequency, rows, n | rows (256) | element | 0 | 8 |
// | rms_norm_rope2_f32        | x, gain, cos, sin, positions, y, x2, gain2, y2; cols, eps, offset, heads, steps, half, interleaved, rows1, eps2, offset2, heads2, n | rows1 + rows2 (256) | element | 7 | 20 |
// | norm_rope_heads_f32       | x, gain, x2, gain2, x3, cos, sin, positions, pos, y, y2, y3; cols, eps, offset, heads, eps2, offset2, heads2, steps, half, interleaved, rows1, rows2, flags, cap, stride, n | rows1 + 2 · rows2 (256) | element | 7 | 20 |
// | topk_candidates_f32       | logits, candv, candi, flags; vocab, rowstride, rowoffset, invt, topk, bpr, n             | rows · bpr (256)                 | score           | 3k + 3 | 4.25 |
// | topk_greedy_f32           | logits, candv, candi, flags, parts, ids, stats, step, counters; vocab, rowstride, rowoffset, invt, topk, bpr, topp, logminp, seed, rows, slots, n | rows · bpr (256) | score | 3k + 7 | 4.25 |
// | sample_rows_f32           | logits, ids, stats, step; vocab, rowstride, rowoffset, invt, topk, topp, logminp, seed, rows, n | rows (256)               | score           | 150   | 36    |
// | sample_candidates_f32     | logits, ids, stats, step, candv, candi, flags; (sample_rows_f32's), slots before n       | rows (256)                       | candidate       | 150   | 36    |
// | sample_candidates_greedy_f32 | logits, ids, stats, step, candv, candi, flags, parts; (sample_candidates_f32's)       | rows (256)                       | candidate       | 150   | 36    |
//
// (w: the widest product's packed words a row; R: the rows a kernel is written for; k: the top-k; bpr: a row's blocks of
// 2,048 scores. penalize_rows_f32 counts the row's copy; its few penalized ids, each over the window, are left out. The
// sampler counts the passes without top-k or top-p (the max, the sum, the draw's two, the entropy and the five
// alternatives, each a weight from an exp, about 16 operations); top-k adds k cheap passes, top-p 25 weighted ones. gemv_multi_finish_f32's act: 0 none, else the gated
// activation's kind + 1, writing hidden = act(y0) · y1 for a gate/up pair launched with one row of blocks.)
internal static partial class GcnKernels
{
    /// <summary>The decoding kernels, in the order they join <see cref="All"/>.</summary>
    internal static IReadOnlyList<GcnKernelSource> Decoding => _decoding ??= BuildDecoding();

    // Built on first use: the text reads this file's and GcnKernels.Products.cs's lists, whose static initializers may
    // come later.
    private static GcnKernelSource[]? _decoding;

    /// <summary>The fused forms of the packed few-row products (<see cref="PackedGemv"/>).</summary>
    internal enum PackedGemvVariant
    {
        Plain,
        Silu,
        Gelu,
        Multi,

        /// <summary>The product, then the residual addition and the next norm (and y's own norm first), by its last block.</summary>
        AddNorm,
    }

    // The fused variants, in the order of their names.
    private static readonly string[] FusedVariantNames = ["silu", "gelu", "multi", "addnorm"];

    /// <summary>The packed formats the fused products are written for (the built-in packed layers' formats).</summary>
    internal static IReadOnlyList<PackedWeights> FusedGemvFormats { get; } = [PackedWeights.Int8, PackedWeights.Int4, PackedWeights.BFloat16];

    // Kernel names by ((format · 3 + variant - 1) · 4 + row-count index), built once: no name is formatted per call.
    private static string[]? _fusedGemvNames;

    private static string[] FusedGemvNames => _fusedGemvNames ??= BuildFusedGemvNames();

    private static string[] BuildFusedGemvNames()
    {
        var names = new List<string>();
        foreach (string prefix in (string[])["int8", "int4", "bf16"])
        {
            foreach (string variant in FusedVariantNames)
            {
                foreach (int rows in PackedGemvRowCounts)
                {
                    names.Add(rows == GemvRows ? $"{prefix}_gemv_{variant}_f32" : string.Create(CultureInfo.InvariantCulture, $"{prefix}_gemv_{variant}_r{rows}_f32"));
                }
            }
        }

        return [.. names];
    }

    /// <summary>The fused few-row product of <paramref name="variant"/> for <paramref name="m"/> rows (1 to 8) of <paramref name="format"/>'s weights (int8, int4 or bfloat16).</summary>
    internal static string FusedGemvKernel(PackedWeights format, PackedGemvVariant variant, int m)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(m, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(m, GemvRows);
        if (variant == PackedGemvVariant.Plain || format == PackedWeights.Float16)
        {
            throw new ArgumentOutOfRangeException(nameof(variant), $"no fused {variant} product of {format} weights");
        }

        var counts = PackedGemvRowCounts;
        int i = 0;
        while (counts[i] < m)
        {
            i++;
        }

        int f = format == PackedWeights.Int8 ? 0 : format == PackedWeights.Int4 ? 1 : 2;
        return FusedGemvNames[(f * FusedVariantNames.Length + (int)variant - 1) * 4 + i];
    }

    // ---- Arguments and packing -------------------------------------------------------------------------------------

    /// <summary>A gated product's arguments: the packed product's (<see cref="PackedGemvArguments"/>, x being the gate), then up at 72.</summary>
    internal static IReadOnlyList<GcnArgument> GatedGemvArguments => _gatedGemvArguments ??= [.. PackedGemvArguments, GcnArgument.Buffer(72, "up")];

    private static GcnArgument[]? _gatedGemvArguments;

    /// <summary>Bytes of a gated product's arguments.</summary>
    internal const int GatedGemvArgumentBytes = 80;

    /// <summary>
    /// A product with the add and norms' arguments: the packed product's (<see cref="PackedGemvArguments"/>; counters
    /// required, at least the column ranges + 1), then residual, sum, gain, normalized and ygain (0 unless flags bit 0),
    /// eps, offset, yeps, yoffset and flags (bit 0: y normalized first, the sandwich block), from 72.
    /// </summary>
    internal static IReadOnlyList<GcnArgument> AddNormGemvArguments => _addNormGemvArguments ??=
    [
        .. PackedGemvArguments, GcnArgument.Buffer(72, "residual"), GcnArgument.Buffer(80, "sum"), GcnArgument.Buffer(88, "gain"), GcnArgument.Buffer(96, "normalized"),
        GcnArgument.Buffer(104, "ygain"), GcnArgument.Value(112, 4, "eps"), GcnArgument.Value(116, 4, "offset"), GcnArgument.Value(120, 4, "yeps"),
        GcnArgument.Value(124, 4, "yoffset"), GcnArgument.Value(128, 4, "flags"),
    ];

    private static GcnArgument[]? _addNormGemvArguments;

    /// <summary>Bytes of a product with the add and norms' arguments.</summary>
    internal const int AddNormGemvArgumentBytes = 132;

    // Where the add and norms' arguments start (the last block reads them alone).
    private const int AddNormTailOffset = 72;

    /// <summary>Bytes of a multi-output product's shared arguments (x, part; m, k, chunk, splits; counters; blocks, nmax).</summary>
    internal const int MultiGemvSharedBytes = 48;

    /// <summary>Bytes of each product's arguments in a multi-output product's (q, s, y, bias, n and the padding to 8).</summary>
    internal const int MultiGemvProductBytes = 40;

    /// <summary>The most products a multi-output product takes (grid z).</summary>
    internal const int MultiGemvProducts = 3;

    /// <summary>
    /// A multi-output product's arguments, CUDA's (RunMany's, with its gate/up pair's): x, part (0: each output written
    /// whole), m, k, chunk, splits, counters (0: the splits and the pair finished by gemv_multi_finish_f32; else at least 3 ·
    /// blocks, each product's column ranges, then the pair's), blocks (the column ranges, grid x), nmax (the widest
    /// product's columns: part's row length), then for products 0 to 2 the packed weights, scales (0: none), output, bias
    /// (0: none) and columns, from 48 + 40j; then act (0: none, else the gated kind + 1: hidden = act(y0) · y1 for a pair
    /// of equal widths) and hidden.
    /// </summary>
    internal static IReadOnlyList<GcnArgument> MultiGemvArguments => _multiGemvArguments ??= BuildMultiGemvArguments();

    private static GcnArgument[]? _multiGemvArguments;

    private static GcnArgument[] BuildMultiGemvArguments()
    {
        var list = new List<GcnArgument>
        {
            GcnArgument.Buffer(0, "x"), GcnArgument.Buffer(8, "part"), GcnArgument.Value(16, 4, "m"), GcnArgument.Value(20, 4, "k"),
            GcnArgument.Value(24, 4, "chunk"), GcnArgument.Value(28, 4, "splits"), GcnArgument.Buffer(32, "counters"), GcnArgument.Value(40, 4, "blocks"),
            GcnArgument.Value(44, 4, "nmax"),
        };
        for (int j = 0; j < MultiGemvProducts; j++)
        {
            int at = MultiGemvSharedBytes + MultiGemvProductBytes * j;
            list.Add(GcnArgument.Buffer(at, $"q{j}"));
            list.Add(GcnArgument.Buffer(at + 8, $"s{j}"));
            list.Add(GcnArgument.Buffer(at + 16, $"y{j}"));
            list.Add(GcnArgument.Buffer(at + 24, $"bias{j}"));
            list.Add(GcnArgument.Value(at + 32, 4, $"n{j}"));
        }

        list.Add(GcnArgument.Value(MultiPairOffset, 4, "act"));
        list.Add(GcnArgument.Buffer(MultiPairOffset + 4, "hidden"));
        return [.. list];
    }

    // Where a multi-output product's pair arguments (act, hidden) start: past the last product's width.
    private const int MultiPairOffset = MultiGemvSharedBytes + MultiGemvProductBytes * (MultiGemvProducts - 1) + 36;

    /// <summary>Bytes of a multi-output product's arguments (hidden ends them).</summary>
    internal const int MultiGemvArgumentBytes = MultiPairOffset + 12;

    /// <summary>
    /// gemv_multi_finish_f32's arguments: part, hidden (0: none), splits, m, nmax, act (0: none, else the gated kind +
    /// 1), then for products 0 to 2 the scales (0: unscaled), output, bias (0: none) and columns, from 32 + 32j.
    /// </summary>
    internal static IReadOnlyList<GcnArgument> MultiFinishArguments => _multiFinishArguments ??= BuildMultiFinishArguments();

    private static GcnArgument[]? _multiFinishArguments;

    private static GcnArgument[] BuildMultiFinishArguments()
    {
        var list = new List<GcnArgument>
        {
            GcnArgument.Buffer(0, "part"), GcnArgument.Buffer(8, "hidden"), GcnArgument.Value(16, 4, "splits"), GcnArgument.Value(20, 4, "m"),
            GcnArgument.Value(24, 4, "nmax"), GcnArgument.Value(28, 4, "act"),
        };
        for (int j = 0; j < MultiGemvProducts; j++)
        {
            int at = MultiFinishSharedBytes + MultiFinishProductBytes * j;
            list.Add(GcnArgument.Buffer(at, $"s{j}"));
            list.Add(GcnArgument.Buffer(at + 8, $"y{j}"));
            list.Add(GcnArgument.Buffer(at + 16, $"bias{j}"));
            list.Add(GcnArgument.Value(at + 24, 4, $"n{j}"));
        }

        return [.. list];
    }

    private const int MultiFinishSharedBytes = 32, MultiFinishProductBytes = 32;

    /// <summary>One product of a multi-output product or its finish: packed weights, scales, output, bias (0 for those absent) and columns.</summary>
    internal readonly record struct MultiProduct(ulong Packed, ulong Scales, ulong Output, ulong Bias, uint Columns);

    /// <summary>A multi-output product's arguments (<see cref="MultiGemvArguments"/>) into <paramref name="arguments"/>, <see cref="MultiGemvArgumentBytes"/> long; products past those given are the first again with 0 columns.</summary>
    internal static void PackMultiGemv(Span<byte> arguments, ulong x, ulong part, uint m, uint k, uint chunk, uint splits, uint blocks, uint nmax,
        ReadOnlySpan<MultiProduct> products, ulong counters = 0, uint act = 0, ulong hidden = 0)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(arguments.Length, MultiGemvArgumentBytes);
        Span<ulong> values = stackalloc ulong[11 + 5 * MultiGemvProducts];
        (values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8]) = (x, part, m, k, chunk, splits, counters, blocks, nmax);
        (values[^2], values[^1]) = (act, hidden);
        for (int j = 0; j < MultiGemvProducts; j++)
        {
            var p = products[j < products.Length ? j : 0];
            values[9 + 5 * j] = p.Packed;
            values[10 + 5 * j] = p.Scales;
            values[11 + 5 * j] = p.Output;
            values[12 + 5 * j] = p.Bias;
            values[13 + 5 * j] = j < products.Length ? p.Columns : 0u;
        }

        PackArguments(MultiGemvArguments, arguments, values);
    }

    /// <summary>Bytes of gemv_multi_finish_f32's arguments.</summary>
    internal const int MultiFinishArgumentBytes = MultiFinishSharedBytes + MultiFinishProductBytes * (MultiGemvProducts - 1) + 28;

    /// <summary>gemv_multi_finish_f32's arguments (<see cref="MultiFinishArguments"/>) into <paramref name="arguments"/>, <see cref="MultiFinishArgumentBytes"/> long.</summary>
    internal static void PackMultiFinish(Span<byte> arguments, ulong part, ulong hidden, uint splits, uint m, uint nmax, uint act, ReadOnlySpan<MultiProduct> products)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(arguments.Length, MultiFinishArgumentBytes);
        Span<ulong> values = stackalloc ulong[6 + 4 * MultiGemvProducts];
        (values[0], values[1], values[2], values[3], values[4], values[5]) = (part, hidden, splits, m, nmax, act);
        for (int j = 0; j < MultiGemvProducts; j++)
        {
            var p = products[j < products.Length ? j : 0];
            values[6 + 4 * j] = p.Scales;
            values[7 + 4 * j] = p.Output;
            values[8 + 4 * j] = p.Bias;
            values[9 + 4 * j] = j < products.Length ? p.Columns : 0u;
        }

        PackArguments(MultiFinishArguments, arguments, values);
    }

    /// <summary>
    /// pow_f32's flags for the exponent <paramref name="p"/> (CUDA's PowFlags, the tests MathF.Pow makes on it): 1 an odd
    /// integer, 2 finite and not an integer, 4 NaN, 8 zero, 16 one, 32 two.
    /// </summary>
    internal static uint PowFlags(float p)
    {
        bool integer = float.IsFinite(p) && MathF.Floor(p) == p;
        uint flags = 0;
        if (integer && MathF.IEEERemainder(p, 2f) != 0f)
        {
            flags |= 1;
        }

        if (float.IsFinite(p) && !integer)
        {
            flags |= 2;
        }

        if (float.IsNaN(p))
        {
            flags |= 4;
        }

        flags |= p == 0f ? 8u : p == 1f ? 16u : p == 2f ? 32u : 0u;
        return flags;
    }

    private static GcnKernelSource[] BuildDecoding()
    {
        var kernels = new List<GcnKernelSource>();
        foreach (var format in FusedGemvFormats)
        {
            foreach (var variant in (PackedGemvVariant[])[PackedGemvVariant.Silu, PackedGemvVariant.Gelu, PackedGemvVariant.Multi, PackedGemvVariant.AddNorm])
            {
                foreach (int rows in PackedGemvRowCounts)
                {
                    kernels.Add(PackedGemv(format, rows, variant));
                }
            }
        }

        kernels.Add(GemvMultiFinish());
        kernels.Add(GemvAddNorm());
        kernels.Add(AttentionRows("attention_rows_f32", AttentionCache.Float32, spans: false, rows: true));
        kernels.Add(AttentionRows("attention_rows_bf16", AttentionCache.BFloat16, spans: false, rows: true));
        kernels.Add(SegmentSpans());
        kernels.Add(ArgMaxRows());
        kernels.Add(ClassMatch());
        kernels.Add(AddRmsNormAffine());
        kernels.Add(NormRope(NormRopeForm.Single));
        kernels.Add(Rope());
        kernels.Add(OneHot());
        kernels.Add(HistoryPush());
        kernels.Add(PenalizeRows());
        kernels.Add(NormRope(NormRopeForm.Pair));
        kernels.Add(NormRope(NormRopeForm.Heads));
        kernels.Add(TopKCandidates(greedy: false));
        kernels.Add(TopKCandidates(greedy: true));
        kernels.Add(Sampler(SamplerForm.Rows));
        kernels.Add(Sampler(SamplerForm.Candidates));
        kernels.Add(Sampler(SamplerForm.Greedy));
        return [.. kernels];
    }

    // ---- The gated products' input rows ---------------------------------------------------------------------------

    /// <summary>
    /// A gated product's input values, made as they are read (see the file's remarks): the lanes' offsets into gate and up,
    /// the activation's text written once for a step of four rows of k and once for a single row, and its constants set
    /// before the loops.
    /// </summary>
    private sealed class GatedInputs
    {
        private readonly ProductWriter _w;
        private readonly string _gate, _up, _stepOffset, _singleOffset, _g, _u, _h, _g1, _u1, _h1;
        private readonly StringBuilder _step = new(), _single = new();
        private readonly int _rows;

        private GatedInputs(ProductWriter w, int rows, string gate, string up)
        {
            (_w, _rows, _gate, _up) = (w, rows, gate, up);
            (_stepOffset, _singleOffset, _g, _u, _h, _g1, _u1, _h1) = (w.V(), w.V(), w.V(), w.V(), w.V(), w.V(), w.V(), w.V());
        }

        /// <summary>
        /// Sets up a gated product's inputs: <paramref name="gate"/> and <paramref name="up"/> point at the split's first
        /// row of k (advanced with the product's own pointer), rows past m read the last.
        /// </summary>
        public static GatedInputs Begin(ProductWriter w, PackedGemvVariant variant, int rows, string gate, string up, string k, string lastRow)
        {
            var inputs = new GatedInputs(w, rows, gate, up);
            string row = w.V(), quarter = w.V(), single = w.V();
            w.L($"v_lshrrev_b32_e32 {row}, 2, {w.Tid}                ; a step: lane l makes row l / 4's value at k + l mod 4");
            w.L($"v_min_u32_e32 {row}, {lastRow}, {row}");
            w.L($"v_mul_lo_u32 {inputs._stepOffset}, {row}, {k}");
            w.L($"v_and_b32_e32 {quarter}, 3, {w.Tid}");
            w.L($"v_add_u32_e32 {inputs._stepOffset}, {inputs._stepOffset}, {quarter}");
            w.L($"v_lshlrev_b32_e32 {inputs._stepOffset}, 2, {inputs._stepOffset}");
            w.L($"v_min_u32_e32 {single}, {lastRow}, {w.Tid}              ; a single row of k: lane l makes row l's");
            w.L($"v_mul_lo_u32 {inputs._singleOffset}, {single}, {k}");
            w.L($"v_lshlrev_b32_e32 {inputs._singleOffset}, 2, {inputs._singleOffset}");

            // The activation's text for both, its constants held from here (registers numbered apart from the product's).
            int kind = variant == PackedGemvVariant.Silu ? 0 : 1;
            var p = new PointwiseWriter(firstVector: 3000, firstScalar: 3000);
            p.Into(inputs._step);
            p.Gated(inputs._h, inputs._g, inputs._u, kind);
            p.Into(inputs._single);
            p.Gated(inputs._h1, inputs._g1, inputs._u1, kind);
            w.Lines(p.Held);
            return inputs;
        }

        /// <summary>A step's values into the scalar registers %s{xs[r] + u} (row r, k + u), as the plain products' scalar loads.</summary>
        public void Step(int[] xs)
        {
            _w.L($"global_load_dword {_g}, {_stepOffset}, {_gate}");
            _w.L($"global_load_dword {_u}, {_stepOffset}, {_up}");
            _w.Lines(_step);
            for (int r = 0; r < _rows; r++)
            {
                for (int u = 0; u < 4; u++)
                {
                    _w.L($"v_readlane_b32 %s{xs[r] + u}, {_h}, {4 * r + u}");
                }
            }
        }

        /// <summary>A single row of k's values into <paramref name="x1"/> (one register a row).</summary>
        public void Single(string[] x1)
        {
            _w.L($"global_load_dword {_g1}, {_singleOffset}, {_gate}");
            _w.L($"global_load_dword {_u1}, {_singleOffset}, {_up}");
            _w.Lines(_single);
            for (int r = 0; r < _rows; r++)
            {
                _w.L($"v_readlane_b32 {x1[r]}, {_h1}, {r}");
            }
        }

        /// <summary>Both pointers on by <paramref name="bytes"/> (the product's pointer moves with them).</summary>
        public void Advance(string bytes) => _w.Add64(_up, bytes);
    }

    // ---- gemv_multi_finish_f32 -------------------------------------------------------------------------------------

    // y_z[i] = Σ_split part[((z · splits + split) · m + row) · nmax + col] in split order (· s_z[col]) (+ bias_z[col]) for
    // i = row · n_z + col, z = workgroup id y; with act, also y_1 and hidden[i] = act(y_0[i]) · y_1[i].
    private static GcnKernelSource GemvMultiFinish()
    {
        var w = new ProductWriter("gemv_multi_finish_f32");
        w.Head(MultiFinishArguments, ExpansionBlock, dispatch: true, workgroupIds: 2, loadDwords: MultiFinishSharedBytes / 4);
        string part = w.Pointer("part"), hidden = w.Pointer("hidden"), splits = w.Scalar("splits"), m = w.Scalar("m"), nmax = w.Scalar("nmax"), act = w.Scalar("act");
        int p0 = w.SRange(7), p1 = w.SRange(7);
        string offset = w.S(), offset1 = w.S(), t = w.S();

        // A product's scales, output, bias and width (28 bytes: the last product's width ends the arguments).
        void LoadProduct(int first, string at)
        {
            string bias = w.S(), width = w.S();
            w.L($"s_add_u32 {bias}, {at}, 16");
            w.L($"s_add_u32 {width}, {at}, 24");
            w.L($"s_load_dwordx4 %s[{first}:{first + 3}], s[2:3], {at}");
            w.L($"s_load_dwordx2 %s[{first + 4}:{first + 5}], s[2:3], {bias}");
            w.L($"s_load_dword %s{first + 6}, s[2:3], {width}");
        }

        w.L($"s_lshl_b32 {offset}, {w.GroupY}, 5");
        w.L($"s_add_u32 {offset}, {offset}, {MultiFinishSharedBytes}            ; this row of blocks' product's arguments");
        LoadProduct(p0, offset);
        w.L($"s_cmp_eq_u32 {act}, 0");
        w.L($"s_cbranch_scc1 .Lone");
        w.L($"s_add_u32 {offset1}, {offset}, {MultiFinishProductBytes}          ; a gate/up pair: the up product's too");
        LoadProduct(p1, offset1);
        w.Label(".Lone");
        string n0 = $"%s{p0 + 6}";
        string count = w.S(), splitElements = w.S();
        w.L($"s_mul_i32 {count}, {m}, {n0}");
        w.L($"s_mul_i32 {splitElements}, {m}, {nmax}");
        string splitBytes = w.Bytes64(splitElements);
        w.L($"s_mul_i32 {t}, {w.GroupY}, {splits}");
        w.L($"s_mul_i32 {t}, {t}, {splitElements}");
        string part0 = w.Offset64(part, t);
        w.L($"s_mul_i32 {t}, {splits}, {splitElements}");
        string part1 = w.Offset64(part0, t);
        string index = w.GridIndex();
        string reciprocal = w.Reciprocal(n0);

        // The pair's activation, one text a kind, its constants held from here.
        var p = new PointwiseWriter(firstVector: 3000, firstScalar: 3000);
        string g = w.V(), u = w.V(), h = w.V();
        var kinds = new StringBuilder[GatedKinds.Length];
        for (int kind = 0; kind < kinds.Length; kind++)
        {
            p.Into(kinds[kind] = new StringBuilder());
            p.Gated(h, g, u, kind);
        }

        w.Lines(p.Held);
        w.GridLoop(count, index, () =>
        {
            var (row, col) = w.DivMod(index, n0, reciprocal);
            string at = w.V(), outAt = w.V(), colAt = w.V();
            w.L($"v_mul_lo_u32 {at}, {row}, {nmax}");
            w.L($"v_add_u32_e32 {at}, {at}, {col}");
            w.L($"v_lshlrev_b32_e32 {at}, 2, {at}                         ; the element's offset in each split's rows");
            w.L($"v_lshlrev_b32_e32 {outAt}, 2, {index}");
            w.L($"v_lshlrev_b32_e32 {colAt}, 2, {col}");
            MultiOutput(w, g, part0, at, splits, splitBytes, $"%s[{p0}:{p0 + 1}]", $"%s[{p0 + 2}:{p0 + 3}]", $"%s[{p0 + 4}:{p0 + 5}]", outAt, colAt, "0");
            w.L($"s_cmp_eq_u32 {act}, 0");
            w.L($"s_cbranch_scc1 .Lnext");
            MultiOutput(w, u, part1, at, splits, splitBytes, $"%s[{p1}:{p1 + 1}]", $"%s[{p1 + 2}:{p1 + 3}]", $"%s[{p1 + 4}:{p1 + 5}]", outAt, colAt, "1");
            for (int kind = 0; kind < kinds.Length; kind++)
            {
                w.L($"s_cmp_eq_u32 {act}, {kind + 1}");
                w.L($"s_cbranch_scc0 .Lkind{kind}");
                w.Lines(kinds[kind]);
                w.L($"global_store_dword {outAt}, {h}, {hidden}");
                w.L($"s_branch .Lnext");
                w.Label($".Lkind{kind}");
            }

            w.Label(".Lnext");
        });
        return w.Source();
    }

    // One product's output element into `value` and its output: the splits' sums in order, the scale, the bias.
    private static void MultiOutput(ProductWriter w, string value, string part, string at, string splits, string splitBytes, string scales, string output, string bias,
        string outAt, string colAt, string label)
    {
        string loaded = w.V(), split = w.S(), from = w.SPair();
        w.L($"v_mov_b32_e32 {value}, 0");
        w.L($"s_mov_b64 {from}, {part}");
        w.L($"s_mov_b32 {split}, 0");
        w.Label($".Lsplits{label}");
        w.L($"s_cmp_ge_u32 {split}, {splits}");
        w.L($"s_cbranch_scc1 .Lsummed{label}");
        w.L($"global_load_dword {loaded}, {at}, {from}");
        w.L($"v_add_f32_e32 {value}, {value}, {loaded}");
        w.Add64Pair(from, splitBytes);
        w.L($"s_add_u32 {split}, {split}, 1");
        w.L($"s_branch .Lsplits{label}");
        w.Label($".Lsummed{label}");
        w.L($"s_cmp_eq_u64 {scales}, 0");
        w.L($"s_cbranch_scc1 .Lscaled{label}");
        string scale = w.V();
        w.L($"global_load_dword {scale}, {colAt}, {scales}");
        w.L($"v_mul_f32_e32 {value}, {value}, {scale}");
        w.Label($".Lscaled{label}");
        w.L($"s_cmp_eq_u64 {bias}, 0");
        w.L($"s_cbranch_scc1 .Lbiased{label}");
        string b = w.V();
        w.L($"global_load_dword {b}, {colAt}, {bias}");
        w.L($"v_add_f32_e32 {value}, {value}, {b}");
        w.Label($".Lbiased{label}");
        w.L($"global_store_dword {outAt}, {value}, {output}");
    }

    // ---- gemv_addnorm_f32 ------------------------------------------------------------------------------------------

    // A packed product's finish with the residual addition and the next norm, a block a row: y = Σ_split part (· s[j]),
    // with flags bit 0 y = RmsNormAffine(y, ygain, yeps, yoffset) (a sandwich block's norm after the projection), then sum
    // = residual + y and normalized = RmsNormAffine(sum, gain, eps, offset); y, sum and normalized written (the CPU's
    // operations in its order, Backend.PackedMatMulNormAddRmsNormKernel). One split: part is y itself and s is 0.
    private static GcnKernelSource GemvAddNorm()
    {
        var w = new RowWriter("gemv_addnorm_f32", arrays: 1);
        var a = w.Arguments(["part", "s", "y", "residual", "sum", "gain", "normalized", "ygain"], ["cols", "splits", "size", "eps", "offset", "yeps", "yoffset", "flags", "n"]);
        string cols = a.Scalar("cols"), splits = a.Scalar("splits"), flags = a.Scalar("flags");
        string rc = w.ReciprocalOf(cols), splitBytes = w.Bytes64(a.Scalar("size"));
        w.BeginRows(a.Scalar("n"));
        int offset = w.RowOffset(cols);
        string pb = w.Base(a.Pointer("part"), offset), yb = w.Base(a.Pointer("y"), offset), rb = w.Base(a.Pointer("residual"), offset);
        string sb = w.Base(a.Pointer("sum"), offset), nb = w.Base(a.Pointer("normalized"), offset);
        string scales = a.Pointer("s"), gain = a.Pointer("gain"), ygain = a.Pointer("ygain");
        string acc = w.V(), at = w.SPair(), split = w.S(), loaded = w.V();
        var vs = w.Vs();
        var ss = w.Vs();
        int uses = 0;

        // Pass 1: y from the splits, Σ y².
        void Pass1(bool scaled)
        {
            w.Columns(cols,
                e => $"global_load_dword {vs[e.K]}, {e.Off}, {pb}{e.Imm}" + (scaled ? $"\nglobal_load_dword {ss[e.K]}, {e.Off}, {scales}{e.Imm}" : ""),
                e =>
                {
                    string l = string.Create(CultureInfo.InvariantCulture, $".Lsplit{uses++}");
                    return $"""
                        s_mov_b64 {at}, {pb}
                        s_mov_b32 {split}, 1
                        {l}:
                        s_cmp_ge_u32 {split}, {splits}
                        s_cbranch_scc1 {l}_end
                        s_add_u32 {RowWriter.Lo(at)}, {RowWriter.Lo(at)}, {RowWriter.Lo(splitBytes)}
                        s_addc_u32 {RowWriter.Hi(at)}, {RowWriter.Hi(at)}, {RowWriter.Hi(splitBytes)}
                        global_load_dword {loaded}, {e.Off}, {at}{e.Imm}
                        v_add_f32_e32 {vs[e.K]}, {vs[e.K]}, {loaded}
                        s_add_u32 {split}, {split}, 1
                        s_branch {l}
                        {l}_end:
                        {(scaled ? $"v_mul_f32_e32 {vs[e.K]}, {vs[e.K]}, {ss[e.K]}" : "")}
                        global_store_dword {e.Off}, {vs[e.K]}, {yb}{e.Imm}
                        v_fma_f32 {acc}, {vs[e.K]}, {vs[e.K]}, {acc}
                        """;
                });
        }

        w.L($"v_mov_b32_e32 {acc}, 0");
        w.L($"s_cmp_eq_u64 {scales}, 0");
        w.L($"s_cbranch_scc0 .Lscaled");
        Pass1(scaled: false);
        w.L($"s_branch .Lsummed");
        w.Label(".Lscaled");
        Pass1(scaled: true);
        w.Label(".Lsummed");
        w.Reduce(RowWriter.Add, acc);
        string ry = w.V();
        w.L($"v_mul_f32_e32 {ry}, {acc}, {rc}");
        w.L($"v_add_f32_e32 {ry}, {a.Scalar("yeps")}, {ry}");
        w.L($"v_rsq_f32_e32 {ry}, {ry}");

        // Pass 2: y normalized (flags bit 0), sum = residual + y, Σ sum².
        string acc2 = w.V(), bit = w.S();
        var rs = w.Vs();
        var gs = w.Vs();
        var sums = w.Vs();
        void Pass2(bool normalize)
        {
            w.Columns(cols,
                e => $"""
                    global_load_dword {vs[e.K]}, {e.Off}, {yb}{e.Imm}
                    global_load_dword {rs[e.K]}, {e.Off}, {rb}{e.Imm}
                    """ + (normalize ? $"\nglobal_load_dword {gs[e.K]}, {e.Off}, {ygain}{e.Imm}" : ""),
                e => (normalize
                    ? $"""
                        v_add_f32_e32 {gs[e.K]}, {a.Scalar("yoffset")}, {gs[e.K]}
                        v_mul_f32_e32 {vs[e.K]}, {vs[e.K]}, {ry}
                        v_mul_f32_e32 {vs[e.K]}, {vs[e.K]}, {gs[e.K]}
                        global_store_dword {e.Off}, {vs[e.K]}, {yb}{e.Imm}

                        """
                    : "") + $"""
                    v_add_f32_e32 {sums[e.K]}, {rs[e.K]}, {vs[e.K]}
                    global_store_dword {e.Off}, {sums[e.K]}, {sb}{e.Imm}
                    v_fma_f32 {acc2}, {sums[e.K]}, {sums[e.K]}, {acc2}
                    """);
        }

        w.L($"v_mov_b32_e32 {acc2}, 0");
        w.L($"s_and_b32 {bit}, {flags}, 1");
        w.L($"s_cmp_eq_u32 {bit}, 0");
        w.L($"s_cbranch_scc1 .Lplain");
        Pass2(normalize: true);
        w.L($"s_branch .Ladded");
        w.Label(".Lplain");
        Pass2(normalize: false);
        w.Label(".Ladded");
        w.Reduce(RowWriter.Add, acc2);
        string r = w.V();
        w.L($"v_mul_f32_e32 {r}, {acc2}, {rc}");
        w.L($"v_add_f32_e32 {r}, {a.Scalar("eps")}, {r}");
        w.L($"v_rsq_f32_e32 {r}, {r}");

        // Pass 3: normalized = sum · r · (gain + offset).
        var outs = w.Vs();
        w.Columns(cols,
            e => $"""
                global_load_dword {sums[e.K]}, {e.Off}, {sb}{e.Imm}
                global_load_dword {gs[e.K]}, {e.Off}, {gain}{e.Imm}
                """,
            e => $"""
                v_add_f32_e32 {gs[e.K]}, {a.Scalar("offset")}, {gs[e.K]}
                v_mul_f32_e32 {outs[e.K]}, {sums[e.K]}, {r}
                v_mul_f32_e32 {outs[e.K]}, {outs[e.K]}, {gs[e.K]}
                global_store_dword {e.Off}, {outs[e.K]}, {nb}{e.Imm}
                """);
        w.EndRows();
        return w.Source();
    }

    // ---- segment_spans_f32 -----------------------------------------------------------------------------------------

    /// <summary>segment_spans_f32's arguments: starts, begins, ends; rows, steps, window, n.</summary>
    internal static IReadOnlyList<GcnArgument> SegmentSpansArguments => _segmentSpansArguments ??=
    [
        GcnArgument.Buffer(0, "starts"), GcnArgument.Buffer(8, "begins"), GcnArgument.Buffer(16, "ends"), GcnArgument.Value(24, 4, "rows"),
        GcnArgument.Value(28, 4, "steps"), GcnArgument.Value(32, 4, "window"), GcnArgument.Value(36, 4, "n"),
    ];

    private static GcnArgument[]? _segmentSpansArguments;

    // Row i of table b (i = index mod rows) of packed sequences: t = i mod steps sees keys [max(starts[b · steps + t], the
    // window's first), t + 1), written as floats (Backend.AttentionSegmented's ranges, attention_spans_f32's tables).
    private static GcnKernelSource SegmentSpans()
    {
        var w = new ProductWriter("segment_spans_f32");
        w.Head(SegmentSpansArguments, ExpansionBlock, dispatch: true);
        string starts = w.Pointer("starts"), begins = w.Pointer("begins"), ends = w.Pointer("ends");
        string rows = w.Scalar("rows"), steps = w.Scalar("steps"), window = w.Scalar("window"), n = w.Scalar("n");
        string index = w.GridIndex(), byRows = w.Reciprocal(rows), bySteps = w.Reciprocal(steps), windowed = w.SPair();
        w.L($"s_cmp_eq_u32 {window}, 0");
        w.L($"s_cselect_b64 {windowed}, 0, -1                       ; a window: its first position counts");
        w.GridLoop(n, index, () =>
        {
            var (table, row) = w.DivMod(index, rows, byRows);
            var (_, t) = w.DivMod(row, steps, bySteps);
            string at = w.V(), start = w.V(), end = w.V(), first = w.V(), begin = w.V(), mask = w.SPair(), outAt = w.V();
            w.L($"v_mul_lo_u32 {at}, {table}, {steps}");
            w.L($"v_add_u32_e32 {at}, {at}, {t}");
            w.L($"v_lshlrev_b32_e32 {at}, 2, {at}");
            w.L($"global_load_dword {start}, {at}, {starts}");
            w.L($"v_add_u32_e32 {end}, 1, {t}");
            w.L($"v_subrev_u32_e32 {first}, {window}, {end}");
            w.L($"v_cmp_lt_u32_e64 {mask}, {window}, {end}");
            w.L($"s_and_b64 vcc, {mask}, {windowed}");
            w.L($"v_cndmask_b32_e32 {first}, 0, {first}, vcc                ; max(0, end - window), 0 without a window");
            w.L($"v_cvt_i32_f32_e32 {start}, {start}");
            w.L($"v_max_i32_e32 {begin}, {start}, {first}");
            w.L($"v_cvt_f32_i32_e32 {begin}, {begin}");
            w.L($"v_cvt_f32_u32_e32 {end}, {end}");
            w.L($"v_lshlrev_b32_e32 {outAt}, 2, {index}");
            w.L($"global_store_dword {outAt}, {begin}, {begins}");
            w.L($"global_store_dword {outAt}, {end}, {ends}");
        });
        return w.Source();
    }

    // ---- The rows of decoding --------------------------------------------------------------------------------------

    // The column of the row's largest value as the CPU's ArgMaxRow finds it (the first of equal values; NaN never; 0 when
    // the first value is NaN or none is found), in a vector register (every lane the same).
    private static string RowArgMax(RowWriter w, string cols, string xb)
    {
        string best = w.V(), bestColumn = w.V(), better = w.SPair(), tie = w.SPair(), smaller = w.SPair();
        var xs = w.Vs();
        w.L($"v_mov_b32_e32 {best}, {RowWriter.NegativeInfinity}");
        w.L($"v_mov_b32_e32 {bestColumn}, -1");
        w.Columns(cols, e => $"global_load_dword {xs[e.K]}, {e.Off}, {xb}{e.Imm}", e => $"""
            v_cmp_gt_f32_e64 {better}, {xs[e.K]}, {best}
            v_cmp_eq_f32_e64 {tie}, {xs[e.K]}, {best}
            v_cmp_lt_u32_e64 {smaller}, {e.Column}, {bestColumn}
            s_and_b64 {tie}, {tie}, {smaller}
            s_or_b64 vcc, {better}, {tie}
            v_cndmask_b32_e32 {best}, {best}, {xs[e.K]}, vcc
            v_cndmask_b32_e32 {bestColumn}, {bestColumn}, {e.Column}, vcc
            """, columns: true);
        w.Reduce(RowWriter.ArgMax, best, bestColumn);
        string first = w.V(), none = w.SPair();
        w.L($"global_load_dword {first}, {w.Zero}, {xb}");
        w.L($"v_cmp_eq_u32_e64 {none}, -1, {bestColumn}");
        w.L($"v_cmp_u_f32_e32 vcc, {first}, {first}                       ; a NaN first value: column 0, as the CPU's");
        w.L($"s_or_b64 vcc, vcc, {none}");
        w.L($"v_cndmask_b32_e64 {bestColumn}, {bestColumn}, 0, vcc");
        return bestColumn;
    }

    // y[row] = the column of the row's largest value (CpuBackend.ArgMax).
    private static GcnKernelSource ArgMaxRows()
    {
        var w = new RowWriter("argmax_f32", arrays: 2);
        var a = w.Arguments(["x", "y"], ["cols", "n"]);
        string cols = a.Scalar("cols");
        w.BeginRows(a.Scalar("n"));
        string xb = w.Base(a.Pointer("x"), w.RowOffset(cols));
        string column = RowArgMax(w, cols, xb), value = w.V();
        w.L($"v_cvt_f32_u32_e32 {value}, {column}");
        w.StoreFirstLane((w.ElementAddress(a.Pointer("y"), w.Row), value));
        w.EndRows();
        return w.Source();
    }

    // y[row] = 1 when the prediction matches the target (one column: both on the same side of their thresholds; more:
    // the same arg-max), else 0 (CpuBackend.ClassMatch).
    private static GcnKernelSource ClassMatch()
    {
        var w = new RowWriter("class_match_f32", arrays: 2);
        var a = w.Arguments(["predictions", "targets", "y"], ["cols", "threshold", "n"]);
        string cols = a.Scalar("cols");
        w.BeginRows(a.Scalar("n"));
        int offset = w.RowOffset(cols);
        string pb = w.Base(a.Pointer("predictions"), offset), tb = w.Base(a.Pointer("targets"), offset), yAt = w.ElementAddress(a.Pointer("y"), w.Row);
        string result = w.V(), p = w.V(), t = w.V(), above = w.SPair(), half = w.SPair();
        w.L($"s_cmp_eq_u32 {cols}, 1");
        w.L($"s_cbranch_scc0 .Lclasses");
        w.L($"global_load_dword {p}, {w.Zero}, {pb}");
        w.L($"global_load_dword {t}, {w.Zero}, {tb}");
        w.L($"v_cmp_ge_f32_e64 {above}, {p}, {a.Scalar("threshold")}");
        w.L($"v_cmp_le_f32_e64 {half}, 0.5, {t}");
        w.L($"s_xor_b64 vcc, {above}, {half}");
        w.L($"v_cndmask_b32_e64 {result}, 1.0, 0, vcc                  ; the same side: 1");
        w.L($"s_branch .Lmatched");
        w.Label(".Lclasses");
        string predicted = RowArgMax(w, cols, pb), target = RowArgMax(w, cols, tb);
        w.L($"v_cmp_eq_u32_e32 vcc, {predicted}, {target}");
        w.L($"v_cndmask_b32_e64 {result}, 0, 1.0, vcc");
        w.Label(".Lmatched");
        w.StoreFirstLane((yAt, result));
        w.EndRows();
        return w.Source();
    }

    // sum = a + b, y = sum's RMS normalization · (gain + offset) per row (CpuBackend.AddRmsNormAffine's order).
    private static GcnKernelSource AddRmsNormAffine()
    {
        var w = new RowWriter("add_rms_norm_affine_f32", arrays: 1);
        var a = w.Arguments(["a", "b", "sum", "gain", "y"], ["cols", "eps", "offset", "n"]);
        string cols = a.Scalar("cols"), rc = w.ReciprocalOf(cols);
        w.BeginRows(a.Scalar("n"));
        int offset = w.RowOffset(cols);
        string ab = w.Base(a.Pointer("a"), offset), bb = w.Base(a.Pointer("b"), offset), sb = w.Base(a.Pointer("sum"), offset), yb = w.Base(a.Pointer("y"), offset);
        string gain = a.Pointer("gain"), acc = w.V();
        var xs = w.Vs();
        var ys = w.Vs();
        var ss = w.Vs();
        w.L($"v_mov_b32_e32 {acc}, 0");
        w.Columns(cols,
            e => $"""
                global_load_dword {xs[e.K]}, {e.Off}, {ab}{e.Imm}
                global_load_dword {ys[e.K]}, {e.Off}, {bb}{e.Imm}
                """,
            e => $"""
                v_add_f32_e32 {ss[e.K]}, {xs[e.K]}, {ys[e.K]}
                global_store_dword {e.Off}, {ss[e.K]}, {sb}{e.Imm}
                v_fma_f32 {acc}, {ss[e.K]}, {ss[e.K]}, {acc}
                """);
        w.Reduce(RowWriter.Add, acc);
        string r = w.V();
        w.L($"v_mul_f32_e32 {r}, {acc}, {rc}");
        w.L($"v_add_f32_e32 {r}, {a.Scalar("eps")}, {r}");
        w.L($"v_rsq_f32_e32 {r}, {r}");
        w.Columns(cols,
            e => $"""
                global_load_dword {ss[e.K]}, {e.Off}, {sb}{e.Imm}
                global_load_dword {xs[e.K]}, {e.Off}, {gain}{e.Imm}
                """,
            e => $"""
                v_add_f32_e32 {xs[e.K]}, {a.Scalar("offset")}, {xs[e.K]}
                v_mul_f32_e32 {ys[e.K]}, {ss[e.K]}, {r}
                v_mul_f32_e32 {ys[e.K]}, {ys[e.K]}, {xs[e.K]}
                global_store_dword {e.Off}, {ys[e.K]}, {yb}{e.Imm}
                """);
        w.EndRows();
        return w.Source();
    }

    /// <summary>The kernels of the RMS norm with the rotary embedding: one tensor, two (queries and keys) and the attention layer's heads.</summary>
    internal enum NormRopeForm
    {
        /// <summary>rms_norm_rope_f32 (Backend.RmsNormRope).</summary>
        Single,

        /// <summary>rms_norm_rope2_f32 (Backend.RmsNormRopePair).</summary>
        Pair,

        /// <summary>norm_rope_heads_f32 (Backend.NormRopeHeads).</summary>
        Heads,
    }

    /// <summary>A norm-and-rope kernel's name.</summary>
    internal static string NormRopeKernel(NormRopeForm form) => form switch
    {
        NormRopeForm.Single => "rms_norm_rope_f32",
        NormRopeForm.Pair => "rms_norm_rope2_f32",
        _ => "norm_rope_heads_f32",
    };

    // CUDA's argument names of each form: its pointers, then its 32-bit values (RowWriter's order).
    private static (string[] Pointers, string[] Scalars) NormRopeNames(NormRopeForm form) => form switch
    {
        NormRopeForm.Single => (["x", "gain", "cos", "sin", "positions", "y"], ["cols", "eps", "offset", "heads", "steps", "half", "interleaved", "n"]),
        NormRopeForm.Pair => (["x", "gain", "cos", "sin", "positions", "y", "x2", "gain2", "y2"],
            ["cols", "eps", "offset", "heads", "steps", "half", "interleaved", "rows1", "eps2", "offset2", "heads2", "n"]),
        _ => (["x", "gain", "x2", "gain2", "x3", "cos", "sin", "positions", "pos", "y", "y2", "y3"],
            ["cols", "eps", "offset", "heads", "eps2", "offset2", "heads2", "steps", "half", "interleaved", "rows1", "rows2", "flags", "cap", "stride", "n"]),
    };

    /// <summary>
    /// A norm-and-rope kernel's arguments, CUDA's: rms_norm_rope_f32's x, gain, cos, sin, positions, y; cols, eps, offset,
    /// heads, steps, half, interleaved, n; rms_norm_rope2_f32's with x2, gain2, y2 after y and rows1, eps2, offset2, heads2
    /// before n; norm_rope_heads_f32's x, gain, x2, gain2, x3, cos, sin, positions, pos, y, y2, y3; cols, eps, offset,
    /// heads, eps2, offset2, heads2, steps, half, interleaved, rows1, rows2, flags, cap, stride, n.
    /// </summary>
    internal static IReadOnlyList<GcnArgument> NormRopeArguments(NormRopeForm form)
    {
        var (pointers, scalars) = NormRopeNames(form);
        return RowArgumentList(pointers, scalars);
    }

    /// <summary>
    /// The arguments a row kernel declares from its pointers' and 32-bit values' names (<see cref="RowWriter.Arguments"/>'s
    /// layout: each pointer 8 bytes from 0, then each value 4 bytes).
    /// </summary>
    private static GcnArgument[] RowArgumentList(string[] pointers, string[] scalars)
    {
        var list = new GcnArgument[pointers.Length + scalars.Length];
        for (int i = 0; i < pointers.Length; i++)
        {
            list[i] = GcnArgument.Buffer(8 * i, pointers[i]);
        }

        for (int i = 0; i < scalars.Length; i++)
        {
            list[pointers.Length + i] = GcnArgument.Value(8 * pointers.Length + 4 * i, 4, scalars[i]);
        }

        return list;
    }

    /// <summary>Bytes of a row kernel's arguments of <paramref name="pointers"/> pointers and <paramref name="scalars"/> 32-bit values.</summary>
    private static int RowArgumentBytes(int pointers, int scalars) => 8 * pointers + 4 * scalars;

    /// <summary>
    /// rms_norm_rope2_f32's arguments (<see cref="NormRopeArguments"/>) into <paramref name="arguments"/>: rows1 rows of x
    /// (gain, y, eps, offset, heads), then rows2 of x2 (gain2, y2, eps2, offset2, heads2), sharing the tables and positions.
    /// </summary>
    internal static void PackRmsNormRopePair(Span<byte> arguments, ulong x, ulong gain, ulong y, uint rows1, float eps, float offset, uint heads, ulong x2, ulong gain2,
        ulong y2, uint rows2, float eps2, float offset2, uint heads2, ulong cos, ulong sin, ulong positions, uint cols, uint steps, uint half, bool interleaved)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(arguments.Length, RowArgumentBytes(9, 12));
        PackArguments(NormRopeArguments(NormRopeForm.Pair), arguments, x, gain, cos, sin, positions, y, x2, gain2, y2, cols, ArgumentBits(eps), ArgumentBits(offset), heads, steps, half,
            interleaved ? 1u : 0u, rows1, ArgumentBits(eps2), ArgumentBits(offset2), heads2, rows1 + rows2);
    }

    /// <summary>
    /// norm_rope_heads_f32's arguments (<see cref="NormRopeArguments"/>) into <paramref name="arguments"/>: flags 1 the
    /// norms (gains given), 2 bfloat16 caches, 4 a cache position (pos); half 0 when there are no tables.
    /// </summary>
    internal static void PackNormRopeHeads(Span<byte> arguments, ulong q, ulong gainQ, ulong k, ulong gainK, ulong v, ulong cos, ulong sin, ulong positions,
        ulong position, ulong yq, ulong yk, ulong yv, uint cols, float epsQ, float offsetQ, uint heads, float epsK, float offsetK, uint kvHeads, uint steps,
        uint half, bool interleaved, uint rows1, uint rows2, uint flags, uint capacity, uint stride)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(arguments.Length, RowArgumentBytes(12, 16));
        PackArguments(NormRopeArguments(NormRopeForm.Heads), arguments, q, gainQ, k, gainK, v, cos, sin, positions, position, yq, yk, yv, cols, ArgumentBits(epsQ), ArgumentBits(offsetQ),
            heads, ArgumentBits(epsK), ArgumentBits(offsetK), kvHeads, steps, half, interleaved ? 1u : 0u, rows1, rows2, flags, capacity, stride, rows1 + 2 * rows2);
    }

    private static uint ArgumentBits(float value) => BitConverter.SingleToUInt32Bits(value);

    // The RMS norm with gain and the rotary embedding, a block a row (CpuBackend.RmsNormRope: RmsNormAffine, then Rope with
    // sign 1 at the position of the row's step, positions[(row / heads) mod steps]; the dimensions past 2 · half normalized
    // alone). A thread rotates the pairs p = tid + 256 i, each read and written by it alone, so y may be x.
    // - rms_norm_rope_f32: the rows of x into y.
    // - rms_norm_rope2_f32 (CUDA's, one launch for queries and keys): rows past rows1 are x2's (gain2, y2, eps2, offset2,
    //   heads2).
    // - norm_rope_heads_f32 (CUDA's): the attention layer's queries, keys and values [batch, steps, heads·cols] in one
    //   launch: rows1 query rows, then rows2 key rows, then rows2 value rows (row = (b · steps + s) · heads + h). With flags
    //   & 1, queries and keys are normalized with their gains; queries and keys are rotated (half pairs; half 0: none);
    //   values are copied (· 1 · 1, exact). Queries go to y [batch, heads, steps, cols]; keys and values to y2 and y3
    //   [batch, heads2, cap, stride] at row s + pos[0] (flags & 4: a cache) or s, as floats or, with flags & 2, as bfloat16
    //   (to nearest even, NaN quieted: kv_write_bf16's and the CPU's rounding).
    private static GcnKernelSource NormRope(NormRopeForm form)
    {
        bool pair = form == NormRopeForm.Pair, heads3 = form == NormRopeForm.Heads;
        var w = new RowWriter(NormRopeKernel(form), arrays: 1);
        var (pointerNames, scalarNames) = NormRopeNames(form);
        var a = w.Arguments(pointerNames, scalarNames);
        string cols = a.Scalar("cols"), half = a.Scalar("half"), steps = a.Scalar("steps");
        string rc = w.ReciprocalOf(cols), bySteps = AttentionReciprocal(w, steps), byHeads = AttentionReciprocal(w, a.Scalar("heads"));
        string byHeads2 = form == NormRopeForm.Single ? "" : AttentionReciprocal(w, a.Scalar("heads2"));
        string shift = w.S(), pairStep = w.S();
        w.L($"s_cmp_lg_u32 {a.Scalar("interleaved")}, 0");
        w.L($"s_cselect_b32 {shift}, 1, 0                              ; interleaved: pair p at 2p and 2p + 1, else p and p + half");
        w.L($"s_cselect_b32 {pairStep}, 1, {half}");

        // norm_rope_heads_f32's per-call values: the cache position (flags & 4), the flags' bits, bfloat16's rounding.
        string flags = heads3 ? a.Scalar("flags") : "", position0 = w.S(), normFlag = w.S(), halfFlag = w.S(), roundBias = w.S();
        if (heads3)
        {
            string bits = w.S(), loaded = w.V();
            w.L($"s_mov_b32 {position0}, 0");
            w.L($"s_and_b32 {bits}, {flags}, 4");
            w.L($"s_cmp_eq_u32 {bits}, 0");
            w.L($"s_cbranch_scc1 .Lno_cache");
            w.L($"s_load_dword {bits}, {a.Pointer("pos")}, 0x0              ; the cache's first free row, a float");
            w.L($"v_cvt_u32_f32_e32 {loaded}, {bits}");
            w.L($"v_readfirstlane_b32 {position0}, {loaded}");
            w.Label(".Lno_cache");
            w.L($"s_and_b32 {normFlag}, {flags}, 1");
            w.L($"s_and_b32 {halfFlag}, {flags}, 2");
            w.L($"s_movk_i32 {roundBias}, 0x7fff");
        }

        w.BeginRows(a.Scalar("n"));

        // The row's tensor: its row in it, its pointers and values (queries, then keys, then values for the heads).
        string local = w.S(), x = w.SPair(), gain = w.SPair(), y = w.SPair(), heads = w.S(), byRowHeads = w.S(), eps = w.S(), offset = w.S();
        string rowHalf = w.S(), normalize = w.S(), bf16 = w.S(), outSteps = w.S(), outFirst = w.S(), outCols = w.S();
        w.L($"""
            s_mov_b32 {local}, {w.Row}
            s_mov_b64 {x}, {a.Pointer("x")}
            s_mov_b64 {gain}, {a.Pointer("gain")}
            s_mov_b64 {y}, {a.Pointer("y")}
            s_mov_b32 {heads}, {a.Scalar("heads")}
            s_mov_b32 {byRowHeads}, {byHeads}
            s_mov_b32 {eps}, {a.Scalar("eps")}
            s_mov_b32 {offset}, {a.Scalar("offset")}
            s_mov_b32 {rowHalf}, {half}
            s_mov_b32 {normalize}, {(heads3 ? normFlag : "1")}
            s_mov_b32 {bf16}, 0
            s_mov_b32 {outSteps}, {steps}
            s_mov_b32 {outFirst}, 0
            s_mov_b32 {outCols}, {cols}
            """);
        if (pair || heads3)
        {
            w.L($"""
                s_cmp_lt_u32 {w.Row}, {a.Scalar("rows1")}
                s_cbranch_scc1 .Ltensor
                s_sub_u32 {local}, {w.Row}, {a.Scalar("rows1")}
                s_mov_b64 {x}, {a.Pointer("x2")}
                s_mov_b64 {gain}, {a.Pointer("gain2")}
                s_mov_b64 {y}, {a.Pointer("y2")}
                s_mov_b32 {heads}, {a.Scalar("heads2")}
                s_mov_b32 {byRowHeads}, {byHeads2}
                s_mov_b32 {eps}, {a.Scalar("eps2")}
                s_mov_b32 {offset}, {a.Scalar("offset2")}
                """);
            if (heads3)
            {
                w.L($"""
                    s_mov_b32 {bf16}, {halfFlag}
                    s_mov_b32 {outSteps}, {a.Scalar("cap")}
                    s_mov_b32 {outFirst}, {position0}
                    s_mov_b32 {outCols}, {a.Scalar("stride")}
                    s_cmp_lt_u32 {local}, {a.Scalar("rows2")}
                    s_cbranch_scc1 .Ltensor
                    s_sub_u32 {local}, {local}, {a.Scalar("rows2")}
                    s_mov_b64 {x}, {a.Pointer("x3")}
                    s_mov_b64 {y}, {a.Pointer("y3")}
                    s_mov_b32 {normalize}, 0
                    s_mov_b32 {rowHalf}, 0
                    """);
            }

            w.Label(".Ltensor");
        }

        // The row's step and head, its input row and its output row (in bytes: floats, or bfloat16 halves).
        var (stepRow, head) = AttentionDivMod(w, local, heads, byRowHeads);
        var (batch, step) = AttentionDivMod(w, stepRow, steps, bySteps);
        string xb = w.Base(x, w.Product64(local, cols)), outRow = w.S(), outShift = w.S(), outBack = w.S(), yb = w.SPair();
        if (heads3)
        {
            w.L($"""
                s_mul_i32 {outRow}, {batch}, {heads}
                s_add_u32 {outRow}, {outRow}, {head}
                s_mul_i32 {outRow}, {outRow}, {outSteps}
                s_add_u32 {outRow}, {outRow}, {outFirst}
                s_add_u32 {outRow}, {outRow}, {step}
                s_mul_i32 {outRow}, {outRow}, {outCols}
                """);
        }
        else
        {
            w.L($"s_mul_i32 {outRow}, {local}, {cols}");
        }

        w.L($"""
            s_cmp_lg_u32 {bf16}, 0
            s_cselect_b32 {outShift}, 1, 2
            s_sub_u32 {outBack}, 32, {outShift}
            s_lshl_b32 {RowWriter.Lo(yb)}, {outRow}, {outShift}
            s_lshr_b32 {RowWriter.Hi(yb)}, {outRow}, {outBack}
            s_add_u32 {RowWriter.Lo(yb)}, {RowWriter.Lo(y)}, {RowWriter.Lo(yb)}
            s_addc_u32 {RowWriter.Hi(yb)}, {RowWriter.Hi(y)}, {RowWriter.Hi(yb)}
            """);

        // The norm's scale (1 without it).
        string r = w.V(), acc = w.V();
        var xs = w.Vs();
        w.L($"v_mov_b32_e32 {r}, 1.0");
        if (heads3)
        {
            w.L($"s_cmp_eq_u32 {normalize}, 0");
            w.L($"s_cbranch_scc1 .Lnormed");
        }

        w.L($"v_mov_b32_e32 {acc}, 0");
        w.Columns(cols, e => $"global_load_dword {xs[e.K]}, {e.Off}, {xb}{e.Imm}", e => $"v_fma_f32 {acc}, {xs[e.K]}, {xs[e.K]}, {acc}");
        w.Reduce(RowWriter.Add, acc);
        w.L($"v_mul_f32_e32 {r}, {acc}, {rc}");
        w.L($"v_add_f32_e32 {r}, {eps}, {r}");
        w.L($"v_rsq_f32_e32 {r}, {r}");
        if (heads3)
        {
            w.Label(".Lnormed");
        }

        // A value's store at column `index` of the output row: a float, or (keys and values with flags & 2) its bfloat16.
        int stores = 0;
        string at = w.V(), odd = w.V(), rounded = w.V(), quiet = w.V();
        string Store(string index, string value)
        {
            if (!heads3)
            {
                return $"""
                    v_lshlrev_b32_e32 {at}, 2, {index}
                    global_store_dword {at}, {value}, {yb}
                    """;
            }

            string n = (stores++).ToString(CultureInfo.InvariantCulture);
            return $"""
                s_cmp_lg_u32 {bf16}, 0
                s_cbranch_scc1 .Lhalf{n}
                v_lshlrev_b32_e32 {at}, 2, {index}
                global_store_dword {at}, {value}, {yb}
                s_branch .Lstored{n}
                .Lhalf{n}:
                v_bfe_u32 {odd}, {value}, 16, 1
                v_add3_u32 {rounded}, {value}, {roundBias}, {odd}
                v_lshrrev_b32_e32 {rounded}, 16, {rounded}
                v_lshrrev_b32_e32 {quiet}, 16, {value}
                v_or_b32_e32 {quiet}, 64, {quiet}
                v_cmp_u_f32_e32 vcc, {value}, {value}
                v_cndmask_b32_e32 {rounded}, {rounded}, {quiet}, vcc
                v_lshlrev_b32_e32 {at}, 1, {index}
                global_store_short {at}, {rounded}, {yb}
                .Lstored{n}:
                """;
        }

        // The gains at the given byte offsets (gain + offset, the loads first), or 1 without the norm.
        string Gains(params (string At, string Gain)[] items)
        {
            var loads = new StringBuilder();
            foreach (var (atOffset, g) in items)
            {
                loads.Append("global_load_dword ").Append(g).Append(", ").Append(atOffset).Append(", ").Append(gain).Append('\n');
            }

            foreach (var (_, g) in items)
            {
                loads.Append("v_add_f32_e32 ").Append(g).Append(", ").Append(offset).Append(", ").Append(g).Append('\n');
            }

            if (!heads3)
            {
                return loads.ToString();
            }

            string n = (stores++).ToString(CultureInfo.InvariantCulture);
            var ones = new StringBuilder();
            foreach (var (_, g) in items)
            {
                ones.Append("v_mov_b32_e32 ").Append(g).Append(", 1.0\n");
            }

            return $"""
                {ones}
                s_cmp_eq_u32 {normalize}, 0
                s_cbranch_scc1 .Lgains{n}
                {loads}
                .Lgains{n}:
                """;
        }

        // The row's position and its rows of the tables (none without pairs to rotate).
        string first = w.S(), pairIndex = w.V(), i = w.V(), j = w.V(), iAt = w.V(), jAt = w.V(), pairAt = w.V(), saved = w.SPair();
        string xi = w.V(), xj = w.V(), gi = w.V(), gj = w.V(), c = w.V(), s = w.V(), t = w.V(), yi = w.V(), yj = w.V();
        w.L($"s_cmp_eq_u32 {rowHalf}, 0");
        w.L($"s_cbranch_scc1 .Lpairs_end");
        string positionAt = w.ElementAddress(a.Pointer("positions"), step), loadedPosition = w.V(), position = w.S(), table = w.S();
        w.L($"global_load_dword {loadedPosition}, {w.Zero}, {positionAt}");
        w.L($"v_cvt_i32_f32_e32 {loadedPosition}, {loadedPosition}");
        w.L($"v_readfirstlane_b32 {position}, {loadedPosition}");
        w.L($"s_mul_i32 {table}, {position}, {half}");
        string cb = w.ElementAddress(a.Pointer("cos"), table), sb = w.ElementAddress(a.Pointer("sin"), table);
        w.L($"s_mov_b32 {first}, 0");
        w.Label(".Lpairs");
        w.L($"""
            s_cmp_ge_u32 {first}, {rowHalf}
            s_cbranch_scc1 .Lpairs_end
            v_add_u32_e32 {pairIndex}, {first}, {w.Tid}
            v_cmp_gt_u32_e32 vcc, {rowHalf}, {pairIndex}
            s_and_saveexec_b64 {saved}, vcc
            v_lshlrev_b32_e32 {i}, {shift}, {pairIndex}
            v_add_u32_e32 {j}, {pairStep}, {i}
            v_lshlrev_b32_e32 {iAt}, 2, {i}
            v_lshlrev_b32_e32 {jAt}, 2, {j}
            v_lshlrev_b32_e32 {pairAt}, 2, {pairIndex}
            global_load_dword {xi}, {iAt}, {xb}
            global_load_dword {xj}, {jAt}, {xb}
            global_load_dword {c}, {pairAt}, {cb}
            global_load_dword {s}, {pairAt}, {sb}
            {Gains((iAt, gi), (jAt, gj))}
            v_mul_f32_e32 {xi}, {xi}, {r}
            v_mul_f32_e32 {xi}, {xi}, {gi}
            v_mul_f32_e32 {xj}, {xj}, {r}
            v_mul_f32_e32 {xj}, {xj}, {gj}
            v_mul_f32_e32 {t}, {xj}, {s}
            v_fma_f32 {yi}, {xi}, {c}, -{t}
            v_mul_f32_e32 {t}, {xi}, {s}
            v_fma_f32 {yj}, {xj}, {c}, {t}
            {Store(i, yi)}
            {Store(j, yj)}
            s_or_b64 exec, exec, {saved}
            s_add_u32 {first}, {first}, 0x{RowThreads:x}
            s_branch .Lpairs
            """);
        w.Label(".Lpairs_end");

        // The dimensions past 2 · half: normalized alone (copied for values).
        string column = w.V(), columnAt = w.V(), value = w.V(), g = w.V();
        w.L($"s_lshl_b32 {first}, {rowHalf}, 1");
        w.Label(".Lrest");
        w.L($"""
            s_cmp_ge_u32 {first}, {cols}
            s_cbranch_scc1 .Lrest_end
            v_add_u32_e32 {column}, {first}, {w.Tid}
            v_cmp_gt_u32_e32 vcc, {cols}, {column}
            s_and_saveexec_b64 {saved}, vcc
            v_lshlrev_b32_e32 {columnAt}, 2, {column}
            global_load_dword {value}, {columnAt}, {xb}
            {Gains((columnAt, g))}
            v_mul_f32_e32 {value}, {value}, {r}
            v_mul_f32_e32 {value}, {value}, {g}
            {Store(column, value)}
            s_or_b64 exec, exec, {saved}
            s_add_u32 {first}, {first}, 0x{RowThreads:x}
            s_branch .Lrest
            """);
        w.Label(".Lrest_end");
        w.EndRows();
        return w.Source();
    }

    /// <summary>rope_f32's arguments, CUDA's: x, y, cos, sin, positions; heads, steps, dim, half, interleaved, sign, n (rows · half).</summary>
    internal static IReadOnlyList<GcnArgument> RopeArguments => _ropeArguments ??=
    [
        GcnArgument.Buffer(0, "x"), GcnArgument.Buffer(8, "y"), GcnArgument.Buffer(16, "cos"), GcnArgument.Buffer(24, "sin"), GcnArgument.Buffer(32, "positions"),
        GcnArgument.Value(40, 4, "heads"), GcnArgument.Value(44, 4, "steps"), GcnArgument.Value(48, 4, "dim"), GcnArgument.Value(52, 4, "half"),
        GcnArgument.Value(56, 4, "interleaved"), GcnArgument.Value(60, 4, "sign"), GcnArgument.Value(64, 4, "n"),
    ];

    private static GcnArgument[]? _ropeArguments;

    // Pair p of row r (a work-item each): rotated by the angle of position positions[(r / heads) mod steps], the sine
    // times sign, as CpuBackend.Rope: y[i] = fma(a, c, -(b · s)), y[j] = fma(b, c, a · s).
    private static GcnKernelSource Rope()
    {
        var w = new ProductWriter("rope_f32");
        w.Head(RopeArguments, ExpansionBlock, dispatch: true);
        string x = w.Pointer("x"), y = w.Pointer("y"), cos = w.Pointer("cos"), sin = w.Pointer("sin"), positions = w.Pointer("positions");
        string heads = w.Scalar("heads"), steps = w.Scalar("steps"), dim = w.Scalar("dim"), half = w.Scalar("half"), sign = w.Scalar("sign"), n = w.Scalar("n");
        string index = w.GridIndex(), byHalf = w.Reciprocal(half), byHeads = w.Reciprocal(heads), bySteps = w.Reciprocal(steps);
        string shift = w.S(), pairStep = w.S();
        w.L($"s_cmp_lg_u32 {w.Scalar("interleaved")}, 0");
        w.L($"s_cselect_b32 {shift}, 1, 0");
        w.L($"s_cselect_b32 {pairStep}, 1, {half}");
        w.GridLoop(n, index, () =>
        {
            var (row, pair) = w.DivMod(index, half, byHalf);
            var (step, _) = w.DivMod(row, heads, byHeads);
            var (_, at) = w.DivMod(step, steps, bySteps);
            string positionAt = w.V(), position = w.V(), tableAt = w.V(), c = w.V(), s = w.V(), i = w.V(), j = w.V(), xi = w.V(), xj = w.V(), t = w.V(), yi = w.V(), yj = w.V();
            w.L($"v_lshlrev_b32_e32 {positionAt}, 2, {at}");
            w.L($"global_load_dword {position}, {positionAt}, {positions}");
            w.L($"v_cvt_i32_f32_e32 {position}, {position}");
            w.L($"v_mul_lo_u32 {tableAt}, {position}, {half}");
            w.L($"v_add_u32_e32 {tableAt}, {tableAt}, {pair}");
            w.L($"v_lshlrev_b32_e32 {tableAt}, 2, {tableAt}");
            w.L($"global_load_dword {c}, {tableAt}, {cos}");
            w.L($"global_load_dword {s}, {tableAt}, {sin}");
            w.L($"v_lshlrev_b32_e32 {i}, {shift}, {pair}");
            w.L($"v_mul_lo_u32 {t}, {row}, {dim}");
            w.L($"v_add_u32_e32 {i}, {t}, {i}");
            w.L($"v_add_u32_e32 {j}, {pairStep}, {i}");
            w.L($"v_lshlrev_b32_e32 {i}, 2, {i}");
            w.L($"v_lshlrev_b32_e32 {j}, 2, {j}");
            w.L($"global_load_dword {xi}, {i}, {x}");
            w.L($"global_load_dword {xj}, {j}, {x}");
            w.L($"v_mul_f32_e32 {s}, {sign}, {s}");
            w.L($"v_mul_f32_e32 {t}, {xj}, {s}");
            w.L($"v_fma_f32 {yi}, {xi}, {c}, -{t}");
            w.L($"v_mul_f32_e32 {t}, {xi}, {s}");
            w.L($"v_fma_f32 {yj}, {xj}, {c}, {t}");
            w.L($"global_store_dword {i}, {yi}, {y}");
            w.L($"global_store_dword {j}, {yj}, {y}");
        });
        return w.Source();
    }

    /// <summary>one_hot_f32's arguments, CUDA's: indices, y; classes, last (classes - 1), n (count · classes).</summary>
    internal static IReadOnlyList<GcnArgument> OneHotArguments => _oneHotArguments ??=
        [GcnArgument.Buffer(0, "indices"), GcnArgument.Buffer(8, "y"), GcnArgument.Value(16, 4, "classes"), GcnArgument.Value(20, 4, "last"), GcnArgument.Value(24, 4, "n")];

    private static GcnArgument[]? _oneHotArguments;

    // y[i · classes + c] = 1 where c is index i's class (clamped to [0, last]), else 0.
    private static GcnKernelSource OneHot()
    {
        var w = new ProductWriter("one_hot_f32");
        w.Head(OneHotArguments, ExpansionBlock, dispatch: true);
        string indices = w.Pointer("indices"), y = w.Pointer("y"), classes = w.Scalar("classes"), last = w.Scalar("last"), n = w.Scalar("n");
        string index = w.GridIndex(), byClasses = w.Reciprocal(classes);
        w.GridLoop(n, index, () =>
        {
            var (item, column) = w.DivMod(index, classes, byClasses);
            string at = w.V(), id = w.V(), value = w.V();
            w.L($"v_lshlrev_b32_e32 {at}, 2, {item}");
            w.L($"global_load_dword {id}, {at}, {indices}");
            w.L($"v_cvt_i32_f32_e32 {id}, {id}");
            w.L($"v_min_i32_e32 {id}, {last}, {id}");
            w.L($"v_max_i32_e32 {id}, 0, {id}");
            w.L($"v_cmp_eq_u32_e32 vcc, {id}, {column}");
            w.L($"v_cndmask_b32_e64 {value}, 0, 1.0, vcc");
            w.L($"v_lshlrev_b32_e32 {at}, 2, {index}");
            w.L($"global_store_dword {at}, {value}, {y}");
        });
        return w.Source();
    }

    /// <summary>history_push_f32's arguments, CUDA's: ids, history, length; capacity, rows, n (rows).</summary>
    internal static IReadOnlyList<GcnArgument> HistoryPushArguments => _historyPushArguments ??=
    [
        GcnArgument.Buffer(0, "ids"), GcnArgument.Buffer(8, "history"), GcnArgument.Buffer(16, "length"), GcnArgument.Value(24, 4, "capacity"),
        GcnArgument.Value(28, 4, "rows"), GcnArgument.Value(32, 4, "n"),
    ];

    private static GcnArgument[]? _historyPushArguments;

    // history[r · capacity + length mod capacity] = ids[r] (the length as the device holds it, a float).
    private static GcnKernelSource HistoryPush()
    {
        var w = new ProductWriter("history_push_f32");
        w.Head(HistoryPushArguments, ExpansionBlock, dispatch: true);
        string ids = w.Pointer("ids"), history = w.Pointer("history"), capacity = w.Scalar("capacity"), n = w.Scalar("n");
        string index = w.GridIndex(), byCapacity = w.Reciprocal(capacity), bits = w.S(), length = w.V();
        w.L($"s_load_dword {bits}, {w.Pointer("length")}, 0x0");
        w.L($"v_cvt_u32_f32_e32 {length}, {bits}");
        var (_, slot) = w.DivMod(length, capacity, byCapacity);
        w.GridLoop(n, index, () =>
        {
            string at = w.V(), id = w.V(), target = w.V();
            w.L($"v_lshlrev_b32_e32 {at}, 2, {index}");
            w.L($"global_load_dword {id}, {at}, {ids}");
            w.L($"v_mul_lo_u32 {target}, {index}, {capacity}");
            w.L($"v_add_u32_e32 {target}, {target}, {slot}");
            w.L($"v_lshlrev_b32_e32 {target}, 2, {target}");
            w.L($"global_store_dword {target}, {id}, {history}");
        });
        return w.Source();
    }

    // Repetition penalties (CpuBackend.PenalizeRows), a block a row: the row's logits copied to work; then each of the
    // last n = min(length, lastn) ids of the row's history (a thread each) whose id was not seen more recently and lies in
    // the vocabulary writes work[id] = (x > 0 ? x / repeat : x · repeat) - presence - frequency · count, x the logit and
    // count its appearances in the window. The copy's stores finish (s_waitcnt) before the barrier, so a penalty's store
    // comes after the copy's of the same element (the block's waves share one cache, CU scope).
    private static GcnKernelSource PenalizeRows()
    {
        var w = new RowWriter("penalize_rows_f32", arrays: 0);
        var a = w.Arguments(["logits", "work", "history", "length"],
            ["vocabulary", "stride", "offset", "capacity", "lastn", "repeat", "presence", "frequency", "rows", "n"]);
        string vocabulary = a.Scalar("vocabulary"), capacity = a.Scalar("capacity"), repeat = a.Scalar("repeat");
        string bits = w.S(), lengthValue = w.V(), length = w.S(), window = w.S(), newest = w.S(), lastSlot = w.S();
        w.L($"s_load_dword {bits}, {a.Pointer("length")}, 0x0");
        w.L($"v_cvt_u32_f32_e32 {lengthValue}, {bits}");
        w.L($"v_readfirstlane_b32 {length}, {lengthValue}");
        w.L($"s_min_u32 {window}, {length}, {a.Scalar("lastn")}             ; the window: the last n ids");
        w.L($"s_sub_u32 {newest}, {length}, 1");
        string slot = w.Remainder(newest, capacity);
        w.L($"s_sub_u32 {lastSlot}, {capacity}, 1");
        w.BeginRows(a.Scalar("n"));
        string t = w.S();
        w.L($"s_mul_i32 {t}, {w.Row}, {a.Scalar("stride")}");
        w.L($"s_add_u32 {t}, {t}, {a.Scalar("offset")}");
        string lb = w.ElementAddress(a.Pointer("logits"), t);
        string t2 = w.S();
        w.L($"s_mul_i32 {t2}, {w.Row}, {vocabulary}");
        string wb = w.ElementAddress(a.Pointer("work"), t2);
        string t3 = w.S();
        w.L($"s_mul_i32 {t3}, {w.Row}, {capacity}");
        string hb = w.ElementAddress(a.Pointer("history"), t3);
        var xs = w.Vs();
        w.Columns(vocabulary, e => $"global_load_dword {xs[e.K]}, {e.Off}, {lb}{e.Imm}", e => $"global_store_dword {e.Off}, {xs[e.K]}, {wb}{e.Imm}");
        w.L($"s_waitcnt vmcnt(0)");
        w.L($"s_barrier");

        string first = w.S(), k = w.V(), slotK = w.V(), wrapped = w.V(), id = w.V(), count = w.V(), seen = w.V(), one = w.V(), saved = w.SPair(), applied = w.SPair();
        string q = w.S(), slotQ = w.S(), previous = w.S(), qAt = w.S(), other = w.S(), otherId = w.V(), earlier = w.SPair(), inside = w.SPair();
        string x = w.V(), at = w.V(), scaled = w.V(), result = w.V(), counted = w.V(), d = w.V();
        w.L($"s_mov_b32 {first}, 0");
        w.Label(".Lids");
        w.L($"""
            s_cmp_ge_u32 {first}, {window}
            s_cbranch_scc1 .Lids_end
            v_add_u32_e32 {k}, {first}, {w.Tid}
            v_cmp_gt_u32_e32 vcc, {window}, {k}
            s_and_saveexec_b64 {saved}, vcc
            v_sub_u32_e32 {slotK}, {slot}, {k}
            v_add_u32_e32 {wrapped}, {capacity}, {slotK}
            v_cmp_gt_i32_e32 vcc, 0, {slotK}
            v_cndmask_b32_e32 {slotK}, {slotK}, {wrapped}, vcc
            v_lshlrev_b32_e32 {slotK}, 2, {slotK}
            global_load_dword {id}, {slotK}, {hb}
            v_cvt_i32_f32_e32 {id}, {id}
            v_mov_b32_e32 {count}, 0
            v_mov_b32_e32 {seen}, 0
            s_mov_b32 {q}, 0
            s_mov_b32 {slotQ}, {slot}
            """);
        w.Label(".Lwindow");
        w.L($"""
            s_cmp_ge_u32 {q}, {window}
            s_cbranch_scc1 .Lwindow_end
            s_lshl_b32 {qAt}, {slotQ}, 2
            s_load_dword {other}, {hb}, {qAt}
            v_cvt_i32_f32_e32 {otherId}, {other}
            v_cmp_eq_u32_e32 vcc, {otherId}, {id}
            v_cndmask_b32_e64 {one}, 0, 1, vcc
            v_add_u32_e32 {count}, {count}, {one}
            v_cmp_lt_u32_e64 {earlier}, {q}, {k}
            s_and_b64 {earlier}, {earlier}, vcc
            v_cndmask_b32_e64 {one}, 0, 1, {earlier}
            v_or_b32_e32 {seen}, {seen}, {one}
            s_sub_u32 {previous}, {slotQ}, 1
            s_cmp_eq_u32 {slotQ}, 0
            s_cselect_b32 {slotQ}, {lastSlot}, {previous}
            s_add_u32 {q}, {q}, 1
            s_branch .Lwindow
            """);
        w.Label(".Lwindow_end");
        string unused = w.SPair(), outer = w.SPair(), ds = w.V(), ns = w.V(), rcp = w.V(), e1 = w.V(), quotient = w.V(), e2 = w.V();
        w.L($"""
            v_cmp_eq_u32_e32 vcc, 0, {seen}
            v_cmp_lt_u32_e64 {inside}, {id}, {vocabulary}
            s_and_b64 {applied}, vcc, {inside}
            s_and_saveexec_b64 {outer}, {applied}
            v_lshlrev_b32_e32 {at}, 2, {id}
            global_load_dword {x}, {at}, {lb}
            v_mov_b32_e32 {d}, {repeat}
            v_div_scale_f32 {ds}, {unused}, {d}, {d}, {x}
            v_div_scale_f32 {ns}, vcc, {x}, {d}, {x}
            v_rcp_f32_e32 {rcp}, {ds}
            v_fma_f32 {e1}, -{ds}, {rcp}, 1.0
            v_fma_f32 {rcp}, {e1}, {rcp}, {rcp}
            v_mul_f32_e32 {quotient}, {ns}, {rcp}
            v_fma_f32 {e2}, -{ds}, {quotient}, {ns}
            v_fma_f32 {quotient}, {e2}, {rcp}, {quotient}
            v_fma_f32 {ds}, -{ds}, {quotient}, {ns}
            v_div_fmas_f32 {ds}, {ds}, {rcp}, {quotient}
            v_div_fixup_f32 {quotient}, {ds}, {d}, {x}
            v_mul_f32_e32 {scaled}, {repeat}, {x}
            v_cmp_lt_f32_e32 vcc, 0, {x}
            v_cndmask_b32_e32 {result}, {scaled}, {quotient}, vcc
            v_subrev_f32_e32 {result}, {a.Scalar("presence")}, {result}
            v_cvt_f32_u32_e32 {counted}, {count}
            v_mul_f32_e32 {counted}, {a.Scalar("frequency")}, {counted}
            v_sub_f32_e32 {result}, {result}, {counted}
            global_store_dword {at}, {result}, {wb}
            s_or_b64 exec, exec, {outer}
            s_or_b64 exec, exec, {saved}
            s_add_u32 {first}, {first}, 0x{RowThreads:x}
            s_branch .Lids
            """);
        w.Label(".Lids_end");
        w.EndRows();
        return w.Source();
    }

    // ---- The sampler ------------------------------------------------------------------------------------------------

    /// <summary>Scores a <c>topk_candidates_f32</c> block takes (8 a thread), as CUDA's CandidateSlice.</summary>
    internal const int CandidateSlice = 2048;

    /// <summary>Candidate slots a <c>topk_candidates_f32</c> block writes, as CUDA's CandidateSlots: the largest top-k the candidates serve.</summary>
    internal const int CandidateSlots = 64;

    /// <summary>A block's flag when its ties overflowed its slots even after the retry: the sampler then walks the whole row.</summary>
    internal const uint CandidateOverflow = 0xFFFFFFFF;

    /// <summary>The sampler's kernels: over the whole row, over the top-k candidates, and over them after the greedy rows' check.</summary>
    internal enum SamplerForm
    {
        /// <summary>sample_rows_f32.</summary>
        Rows,

        /// <summary>sample_candidates_f32.</summary>
        Candidates,

        /// <summary>sample_candidates_greedy_f32.</summary>
        Greedy,
    }

    /// <summary>A sampler kernel's name.</summary>
    internal static string SamplerKernel(SamplerForm form) => form switch
    {
        SamplerForm.Rows => "sample_rows_f32",
        SamplerForm.Candidates => "sample_candidates_f32",
        _ => "sample_candidates_greedy_f32",
    };

    // CUDA's argument names of each sampler kernel (with parts in the greedy one where CUDA has pending; see the remarks).
    private static (string[] Pointers, string[] Scalars) SamplerNames(SamplerForm form) => form switch
    {
        SamplerForm.Rows => (["logits", "ids", "stats", "step"], ["vocab", "rowstride", "rowoffset", "invt", "topk", "topp", "logminp", "seed", "rows", "n"]),
        SamplerForm.Candidates => (["logits", "ids", "stats", "step", "candv", "candi", "flags"],
            ["vocab", "rowstride", "rowoffset", "invt", "topk", "topp", "logminp", "seed", "rows", "slots", "n"]),
        _ => (["logits", "ids", "stats", "step", "candv", "candi", "flags", "parts"],
            ["vocab", "rowstride", "rowoffset", "invt", "topk", "topp", "logminp", "seed", "rows", "slots", "n"]),
    };

    // The candidate kernels' argument names (topk_greedy_f32 adds parts and the sampler's: its last block of a row samples it).
    private static (string[] Pointers, string[] Scalars) CandidateNames(bool greedy) => greedy
        ? (["logits", "candv", "candi", "flags", "parts", "ids", "stats", "step", "counters"],
            ["vocab", "rowstride", "rowoffset", "invt", "topk", "bpr", "topp", "logminp", "seed", "rows", "slots", "n"])
        : (["logits", "candv", "candi", "flags"], ["vocab", "rowstride", "rowoffset", "invt", "topk", "bpr", "n"]);

    /// <summary>
    /// A sampler kernel's arguments: CUDA's sample_rows_f32 (logits, ids, stats, step; vocab, rowstride, rowoffset, invt,
    /// topk, topp, then ln(min-p) where CUDA has min-p, seed, rows, n), sample_candidates_f32 (candv, candi and flags after
    /// step, slots before n) and sample_candidates_greedy_f32 (parts after flags, where CUDA has pending).
    /// </summary>
    internal static IReadOnlyList<GcnArgument> SamplerArguments(SamplerForm form)
    {
        var (pointers, scalars) = SamplerNames(form);
        return RowArgumentList(pointers, scalars);
    }

    /// <summary>
    /// A candidate kernel's arguments: CUDA's topk_candidates_f32 (logits, candv, candi, flags; vocab, rowstride, rowoffset,
    /// invt, topk, bpr, n); topk_greedy_f32 adds parts, ids, stats, step and counters after flags (counters 0: the rows are
    /// sampled by sample_candidates_greedy_f32 in a second launch; else at least one a row) and the sampler's topp,
    /// logminp, seed, rows and slots before n.
    /// </summary>
    internal static IReadOnlyList<GcnArgument> CandidateArguments(bool greedy)
    {
        var (pointers, scalars) = CandidateNames(greedy);
        return RowArgumentList(pointers, scalars);
    }

    /// <summary>
    /// The sampler's settings as its kernels take them: the inverse temperature (the CPU's 1 / max(temperature, 1e-3)) and
    /// ln(min-p) (the CPU's MathF.Log, −∞ when min-p is not above 0), so the cut-offs are the CPU's bit for bit.
    /// </summary>
    internal readonly record struct SamplerSettings(uint Vocabulary, uint RowStride, uint RowOffset, float InverseTemperature, uint TopK, float TopP, float LogMinP,
        uint Seed, uint Rows)
    {
        /// <summary>The settings of a call of Backend.SampleRows.</summary>
        public static SamplerSettings Of(int vocabulary, int rowStride, int rowOffset, float temperature, int topK, float topP, float minP, uint seed, int rows) =>
            new((uint)vocabulary, (uint)rowStride, (uint)rowOffset, 1f / MathF.Max(temperature, 1e-3f), (uint)Math.Max(0, topK), topP,
                minP > 0f ? MathF.Log(minP) : float.NegativeInfinity, seed, (uint)rows);
    }

    /// <summary>
    /// A sampler kernel's arguments (<see cref="SamplerArguments"/>) into <paramref name="arguments"/>; <paramref name="candv"/>,
    /// <paramref name="candi"/>, <paramref name="flags"/>, <paramref name="parts"/> and <paramref name="slots"/> only where the form takes them.
    /// </summary>
    internal static void PackSampler(Span<byte> arguments, SamplerForm form, ulong logits, ulong ids, ulong stats, ulong step, in SamplerSettings s, ulong candv = 0,
        ulong candi = 0, ulong flags = 0, ulong parts = 0, uint slots = 0)
    {
        var list = SamplerArguments(form);
        ArgumentOutOfRangeException.ThrowIfNotEqual(arguments.Length, list[^1].Offset + 4);
        ulong[] settings = [s.Vocabulary, s.RowStride, s.RowOffset, ArgumentBits(s.InverseTemperature), s.TopK, ArgumentBits(s.TopP), ArgumentBits(s.LogMinP), s.Seed, s.Rows];
        ulong[] values = form switch
        {
            SamplerForm.Rows => [logits, ids, stats, step, .. settings, s.Rows],
            SamplerForm.Candidates => [logits, ids, stats, step, candv, candi, flags, .. settings, slots, s.Rows],
            _ => [logits, ids, stats, step, candv, candi, flags, parts, .. settings, slots, s.Rows],
        };
        PackArguments(list, arguments, values);
    }

    /// <summary>
    /// A candidate kernel's arguments (<see cref="CandidateArguments"/>) into <paramref name="arguments"/>, for the settings'
    /// rows of <paramref name="blocks"/> blocks each; the greedy kernel's sampling arguments (ids, stats, step, counters)
    /// only where it takes them.
    /// </summary>
    internal static void PackCandidates(Span<byte> arguments, bool greedy, ulong logits, ulong candv, ulong candi, ulong flags, ulong parts, in SamplerSettings s, uint blocks,
        ulong ids = 0, ulong stats = 0, ulong step = 0, ulong counters = 0)
    {
        var list = CandidateArguments(greedy);
        ArgumentOutOfRangeException.ThrowIfNotEqual(arguments.Length, list[^1].Offset + 4);
        ulong[] settings = [s.Vocabulary, s.RowStride, s.RowOffset, ArgumentBits(s.InverseTemperature), s.TopK, blocks];
        PackArguments(list, arguments, greedy
            ? [logits, candv, candi, flags, parts, ids, stats, step, counters, .. settings, ArgumentBits(s.TopP), ArgumentBits(s.LogMinP), s.Seed, s.Rows,
                (ulong)blocks * CandidateSlots, s.Rows * blocks]
            : [logits, candv, candi, flags, .. settings, s.Rows * blocks]);
    }

    // RowWriter's reductions the sampler adds: the larger unsigned value (the candidates' flags); a sum and a minimum.
    private static string UnsignedMax(RowWriter writer, string[] held, string[] other) => $"v_max_u32_e32 {held[0]}, {held[0]}, {other[0]}";

    private static string SumAndMin(RowWriter writer, string[] held, string[] other) => $"""
        v_add_f32_e32 {held[0]}, {held[0]}, {other[0]}
        v_min_f32_e32 {held[1]}, {held[1]}, {other[1]}
        """;

    // The bytes of LDS a sampler kernel takes past its reductions': each thread's part of the draw, then four values.
    private const int SamplerExtraLds = 4 * RowThreads + 16;

    // The sampler, a block a row (Backend.SampleRowsKernel, CpuBackend.SampleRows step by step; each pass over the row's
    // scores is a strided column loop and a reduction in RowWriter's fixed tree, rule 86):
    // 1. The scores: score = logit · invt (with candidates, the candidate scores as they are, already scaled). The row's
    //    max; a +∞ max makes every weight 1 for the +∞ scores and 0 for the rest, a -∞ one (nothing finite) every weight 0,
    //    and neither applies the cut-offs (the CPU's rule for scores that are not finite).
    // 2. The cut-off: top-k's k-th largest distinct score (k passes, each the largest below the last; fewer distinct
    //    scores: -∞), then max(it, max + ln(min-p)) (ln(min-p) the host's, -∞ without min-p), then top-p's bisection:
    //    goal = Σ weights · top-p, from lo = max(max - 40, cut-off), hi = max, 24 halvings of (lo + hi) · 0.5 keeping lo
    //    while the weights at or above the midpoint hold the goal. A weight is exp(score - max) (LLVM's exp, as
    //    GcnKernels.Pointwise.cs) for a score at or above the cut-off and above -∞, else 0.
    // 3. The draw: target = uniform(seed, step, row) · Σ weights, the CPU's counter-based value bit for bit (the MurmurHash3
    //    finalizer of seed ^ step · 0x9E3779B9 ^ row · 0x85EBCA6B, its top 24 bits · 2^-24; step read from device memory
    //    as the CPU reads it). Each thread sums the weights of its contiguous chunk of the row (in order); thread 0 walks
    //    the chunks' sums in order to the first whose running total passes the target (or the last with any weight); that
    //    chunk's thread walks it to the token where the running total passes the target (or its last weighted one): the
    //    CPU's walk, its sums taken by chunks.
    // 4. The statistics (thread 0): the token (ids gets max(token, 0)), its probability, the entropy (Σ -p · log2 p over the
    //    weighted tokens, p = weight / Σ weights), then the five largest weights (five passes, each the largest not taken,
    //    the lower id on ties: RowWriter's arg-max tree) with their probabilities; -1 and 0 past the weighted tokens.
    // With candidates (sample_candidates_f32), the row's scores are its blocks' candidates (topk_candidates_f32), the token
    // ids read through candi; the whole row instead when a block's ties overflowed its slots (its flag 0xFFFFFFFF, the
    // largest of the row's flags) or when a block dropped tied scores at or above the top-k cut-off the candidates give
    // (its flag the dropped score's order-preserving bits). sample_candidates_greedy_f32 first finishes a top-k 1 row from
    // its blocks' parts (topk_greedy_f32): one best score, finite and |s| ≤ 1e38, gives the token with probability 1,
    // entropy 0 and four alternatives of -1 and 0, the statistics the sampler writes for it; any other row samples as
    // sample_candidates_f32 does.
    // The sampler's constants, set before its rows (Sampler, and topk_greedy_f32 when it samples its rows' last blocks).
    private sealed record SamplerConstants(string Rows, string TopK, string Extra, int SlotBase, string D, string Ex, string Sc, string E, string One, string DivN,
        string DivD, string DivQ, StringBuilder ExpText, StringBuilder DivText, string PosInf, string Two32, string ThirtyTwo, string Tid4, string StepNumber,
        string StreamBase, string Slots, string Blocks);

    private static GcnKernelSource Sampler(SamplerForm form)
    {
        var w = new RowWriter(SamplerKernel(form), arrays: 2, extraLds: SamplerExtraLds);
        var (pointerNames, scalarNames) = SamplerNames(form);
        var a = w.Arguments(pointerNames, scalarNames);
        var constants = SamplerSetup(w, a, form);
        w.BeginRows(a.Scalar("n"));
        SamplerRow(w, a, form, constants, w.Row, ".L", glc: false);
        w.EndRows();
        return w.Source();
    }

    // The per-call part of the sampler (rule 85): the held constants of its exp and division, the step number and the
    // random stream's base, the candidate blocks of a row.
    private static SamplerConstants SamplerSetup(RowWriter w, RowWriter.Args a, SamplerForm form)
    {
        bool candidates = form != SamplerForm.Rows, greedy = form == SamplerForm.Greedy;
        string rows = a.Scalar("rows"), topk = a.Scalar("topk"), extra = w.ExtraLdsBase.ToString(CultureInfo.InvariantCulture);
        int slotBase = w.ExtraLdsBase + 4 * RowThreads;

        // The math the passes repeat, written once over fixed registers (LLVM's exp, the division), its constants held.
        var p = new PointwiseWriter(firstVector: 3000, firstScalar: 3000);
        string d = w.V(), ex = w.V(), sc = w.V(), e = w.V(), one = w.V(), divN = w.V(), divD = w.V(), divQ = w.V();
        var expText = new StringBuilder();
        p.Into(expText);
        p.Exp(ex, d);
        var divText = new StringBuilder();
        p.Into(divText);
        p.Div(divQ, divN, divD);
        w.L($"{p.Held}");
        string posInf = w.V(), two32 = w.V(), thirtyTwo = w.V(), tid4 = w.V();
        if (greedy)
        {
            w.L($"v_mov_b32_e32 {posInf}, {RowWriter.PositiveInfinity}");
        }

        w.L($"""
            v_mov_b32_e32 {two32}, 0x4f800000
            v_mov_b32_e32 {thirtyTwo}, 0x42000000
            v_lshlrev_b32_e32 {tid4}, 2, {w.Tid}
            """);

        // The step number (a float in device memory, as the CPU reads it) and the random stream's per-call part.
        string stepNumber = w.S(), streamBase = w.S(), stepBits = w.S(), stepValue = w.V();
        w.L($"""
            s_load_dword {stepBits}, {a.Pointer("step")}, 0x0
            v_cvt_u32_f32_e32 {stepValue}, {stepBits}
            v_readfirstlane_b32 {stepNumber}, {stepValue}
            s_mul_i32 {streamBase}, {stepNumber}, 0x9e3779b9
            s_xor_b32 {streamBase}, {streamBase}, {a.Scalar("seed")}
            """);
        string slots = candidates ? a.Scalar("slots") : "", blocks = w.S();
        if (candidates)
        {
            w.L($"s_lshr_b32 {blocks}, {slots}, 6                          ; the candidate blocks of a row");
        }

        return new(rows, topk, extra, slotBase, d, ex, sc, e, one, divN, divD, divQ, expText, divText, posInf, two32, thirtyTwo, tid4, stepNumber, streamBase, slots, blocks);
    }

    // One row of the sampler (see Sampler's remarks), row `row` (a scalar register), its labels prefixed; with glc,
    // what it reads from the candidate blocks (their candidates, flags and parts) passes the vector L1 (topk_greedy_f32's
    // last block of the row: the other blocks stored them).
    private static void SamplerRow(RowWriter w, RowWriter.Args a, SamplerForm form, SamplerConstants k, string row, string prefix, bool glc)
    {
        bool candidates = form != SamplerForm.Rows, greedy = form == SamplerForm.Greedy;
        string coherent = glc ? " glc" : "";
        var (rows, topk, extra, slotBase, d, ex, sc, e, one, divN) = (k.Rows, k.TopK, k.Extra, k.SlotBase, k.D, k.Ex, k.Sc, k.E, k.One, k.DivN);
        var (divD, divQ, expText, divText, posInf, two32, thirtyTwo, tid4, stepNumber, streamBase) =
            (k.DivD, k.DivQ, k.ExpText, k.DivText, k.PosInf, k.Two32, k.ThirtyTwo, k.Tid4, k.StepNumber, k.StreamBase);
        var (slots, blocks) = (k.Slots, k.Blocks);

        // The row's statistics, its id, its logits and (with candidates) its candidates.
        string statsRow, idsAt = w.ElementAddress(a.Pointer("ids"), row), t = w.S();
        w.L($"""
            s_mul_i32 {t}, {stepNumber}, {rows}
            s_add_u32 {t}, {t}, {row}
            s_mul_i32 {t}, {t}, 13
            """);
        statsRow = w.ElementAddress(a.Pointer("stats"), t);
        string first = w.S();
        w.L($"s_mul_i32 {first}, {row}, {a.Scalar("rowstride")}");
        w.L($"s_add_u32 {first}, {first}, {a.Scalar("rowoffset")}");
        string logitsRow = w.ElementAddress(a.Pointer("logits"), first);
        string candvRow = "", candiRow = "", full = w.S(), rowFlag = w.S();
        if (candidates)
        {
            int at = w.Product64(row, slots);
            candvRow = w.Base(a.Pointer("candv"), at);
            candiRow = w.Base(a.Pointer("candi"), at);
        }

        if (greedy)
        {
            // A top-k 1 row with one best score: its token and statistics from its blocks' parts.
            string partsFirst = w.S(), i = w.S(), best = w.V(), count = w.V(), firstId = w.V(), index = w.V(), offset = w.V(), partsSaved = w.SPair();
            string score = w.V(), ties = w.V(), id = w.V(), tie = w.V(), ided = w.V();
            w.L($"s_mul_i32 {partsFirst}, {row}, {blocks}");
            w.L($"s_mul_i32 {partsFirst}, {partsFirst}, 3");
            string partsRow = w.ElementAddress(a.Pointer("parts"), partsFirst);
            void Parts(string label, string body)
            {
                w.L($"s_mov_b32 {i}, 0");
                w.Label(label);
                w.L($"""
                    s_cmp_ge_u32 {i}, {blocks}
                    s_cbranch_scc1 {label}_end
                    v_add_u32_e32 {index}, {i}, {w.Tid}
                    v_cmp_gt_u32_e32 vcc, {blocks}, {index}
                    s_and_saveexec_b64 {partsSaved}, vcc
                    v_mul_u32_u24_e32 {offset}, 12, {index}
                    {body}
                    s_or_b64 exec, exec, {partsSaved}
                    s_add_u32 {i}, {i}, 0x{RowThreads:x}
                    s_branch {label}
                    """);
                w.Label(label + "_end");
            }

            w.L($"v_mov_b32_e32 {best}, {RowWriter.NegativeInfinity}");
            Parts(prefix + "parts_best", $"""
                global_load_dword {score}, {offset}, {partsRow}{coherent}
                v_max_f32_e32 {best}, {best}, {score}
                """);
            w.Reduce(RowWriter.Max, best);
            w.L($"v_mov_b32_e32 {count}, 0");
            w.L($"v_mov_b32_e32 {firstId}, {RowWriter.PositiveInfinity}");
            Parts(prefix + "parts_ties", $"""
                global_load_dword {score}, {offset}, {partsRow}{coherent}
                global_load_dword {ties}, {offset}, {partsRow} offset:4{coherent}
                global_load_dword {id}, {offset}, {partsRow} offset:8{coherent}
                v_cmp_eq_f32_e32 vcc, {score}, {best}
                v_cndmask_b32_e32 {tie}, 0, {ties}, vcc
                v_add_f32_e32 {count}, {count}, {tie}
                v_cndmask_b32_e32 {ided}, {posInf}, {id}, vcc
                v_min_f32_e32 {firstId}, {firstId}, {ided}
                """);
            w.Reduce(SumAndMin, count, firstId);
            string magnitude = w.V(), single = w.SPair(), minus = w.V(), unity = w.V(), greedySaved = w.SPair();
            w.L($"""
                v_and_b32_e32 {magnitude}, 0x7fffffff, {best}
                v_cmp_ge_f32_e32 vcc, {PointwiseWriter.Literal(1e38f)}, {magnitude}
                v_cmp_eq_f32_e64 {single}, 1.0, {count}
                s_and_b64 vcc, vcc, {single}
                s_cbranch_vccz {prefix}sample
                v_mov_b32_e32 {minus}, -1.0
                v_mov_b32_e32 {unity}, 1.0
                v_cmp_eq_u32_e32 vcc, 0, {w.Tid}
                s_and_saveexec_b64 {greedySaved}, vcc
                global_store_dword {w.Zero}, {firstId}, {idsAt}
                global_store_dword {w.Zero}, {firstId}, {statsRow}
                global_store_dword {w.Zero}, {unity}, {statsRow} offset:4
                global_store_dword {w.Zero}, {w.Zero}, {statsRow} offset:8
                global_store_dword {w.Zero}, {firstId}, {statsRow} offset:12
                global_store_dword {w.Zero}, {unity}, {statsRow} offset:16
                global_store_dword {w.Zero}, {minus}, {statsRow} offset:20
                global_store_dword {w.Zero}, {w.Zero}, {statsRow} offset:24
                global_store_dword {w.Zero}, {minus}, {statsRow} offset:28
                global_store_dword {w.Zero}, {w.Zero}, {statsRow} offset:32
                global_store_dword {w.Zero}, {minus}, {statsRow} offset:36
                global_store_dword {w.Zero}, {w.Zero}, {statsRow} offset:40
                global_store_dword {w.Zero}, {minus}, {statsRow} offset:44
                global_store_dword {w.Zero}, {w.Zero}, {statsRow} offset:48
                s_or_b64 exec, exec, {greedySaved}
                s_branch {prefix}row_done
                """);
            w.Label(prefix + "sample");
        }

        if (candidates)
        {
            // The row's flag: the largest of its blocks' (0xFFFFFFFF, an overflow, wins); an overflow walks the whole row.
            string flagsFirst = w.S(), flag = w.V();
            var fs = w.Vs();
            w.L($"s_mul_i32 {flagsFirst}, {row}, {blocks}");
            string flagsRow = w.ElementAddress(a.Pointer("flags"), flagsFirst);
            w.L($"v_mov_b32_e32 {flag}, 0");
            w.Columns(blocks, el => $"global_load_dword {fs[el.K]}, {el.Off}, {flagsRow}{el.Imm}{coherent}", el => $"v_max_u32_e32 {flag}, {flag}, {fs[el.K]}");
            w.Reduce(UnsignedMax, flag);
            w.L($"""
                v_readfirstlane_b32 {rowFlag}, {flag}
                s_cmp_eq_u32 {rowFlag}, -1
                s_cselect_b32 {full}, 1, 0
                """);
        }
        else
        {
            w.L($"s_mov_b32 {full}, 1");
        }

        // The scores walked: the row's logits (· invt), or its candidates (as they are, their ids through candi).
        string source = w.SPair(), count2 = w.S(), factor = w.S(), mapped = w.S();
        w.Label(prefix + "source");
        w.L($"""
            s_mov_b64 {source}, {logitsRow}
            s_mov_b32 {count2}, {a.Scalar("vocab")}
            s_mov_b32 {factor}, {a.Scalar("invt")}
            s_mov_b32 {mapped}, 0
            """);
        if (candidates)
        {
            w.L($"""
                s_cmp_lg_u32 {full}, 0
                s_cbranch_scc1 {prefix}sourced
                s_mov_b64 {source}, {candvRow}
                s_mov_b32 {count2}, {slots}
                s_mov_b32 {factor}, 1.0
                s_mov_b32 {mapped}, 1
                """);
            w.Label(prefix + "sourced");
        }

        string cnt = count2;
        var xs = w.Vs();
        string Load(RowWriter.Element el) => $"global_load_dword {xs[el.K]}, {el.Off}, {source}{el.Imm}{coherent}";
        string Score(string x) => $"v_mul_f32_e32 {sc}, {factor}, {x}";

        // The weight of the score in sc into e (see the remarks).
        string max = w.V(), threshold = w.V(), infinite = w.SPair();
        string weight = $"""
            v_sub_f32_e32 {d}, {sc}, {max}
            {expText}
            v_cmp_le_f32_e32 vcc, {threshold}, {sc}
            v_cndmask_b32_e32 {e}, 0, {ex}, vcc
            v_cmp_lt_f32_e32 vcc, {RowWriter.NegativeInfinity}, {sc}
            v_cndmask_b32_e32 {e}, 0, {e}, vcc
            v_cmp_eq_f32_e32 vcc, {RowWriter.PositiveInfinity}, {sc}
            v_cndmask_b32_e64 {one}, 0, 1.0, vcc
            v_cndmask_b32_e64 {e}, {e}, {one}, {infinite}
            """;

        // 1. The row's max, and whether it is infinite (+∞: the +∞ tokens alone; -∞: nothing to sample).
        string maxBits = w.S(), magnitudeBits = w.S(), special = w.S();
        w.L($"v_mov_b32_e32 {max}, {RowWriter.NegativeInfinity}");
        w.Columns(cnt, Load, el => $"""
            {Score(xs[el.K])}
            v_max_f32_e32 {max}, {max}, {sc}
            """);
        w.Reduce(RowWriter.Max, max);
        w.L($"""
            v_readfirstlane_b32 {maxBits}, {max}
            s_cmp_eq_u32 {maxBits}, {RowWriter.PositiveInfinity}
            s_cselect_b64 {infinite}, -1, 0
            s_and_b32 {magnitudeBits}, {maxBits}, 0x7fffffff
            s_cmp_eq_u32 {magnitudeBits}, {RowWriter.PositiveInfinity}
            s_cselect_b32 {special}, 1, 0
            v_mov_b32_e32 {threshold}, {RowWriter.NegativeInfinity}
            """);

        // 2. Top-k: k passes, each the largest score below the last (fewer distinct scores than k: -∞).
        string kk = w.S(), largest = w.V(), below = w.SPair(), largestBits = w.S();
        w.L($"""
            s_cmp_lg_u32 {special}, 0
            s_cbranch_scc1 {prefix}topk_done
            s_cmp_eq_u32 {topk}, 0
            s_cbranch_scc1 {prefix}topk_done
            s_cmp_ge_u32 {topk}, {cnt}
            s_cbranch_scc1 {prefix}topk_done
            v_mov_b32_e32 {threshold}, {RowWriter.PositiveInfinity}
            s_mov_b32 {kk}, 0
            """);
        w.Label(prefix + "topk");
        w.L($"""
            s_cmp_ge_u32 {kk}, {topk}
            s_cbranch_scc1 {prefix}topk_done
            v_mov_b32_e32 {largest}, {RowWriter.NegativeInfinity}
            """);
        w.Columns(cnt, Load, el => $"""
            {Score(xs[el.K])}
            v_cmp_gt_f32_e32 vcc, {threshold}, {sc}
            v_cmp_lt_f32_e64 {below}, {largest}, {sc}
            s_and_b64 vcc, vcc, {below}
            v_cndmask_b32_e32 {largest}, {largest}, {sc}, vcc
            """);
        w.Reduce(RowWriter.Max, largest);
        w.L($"""
            v_mov_b32_e32 {threshold}, {largest}
            v_readfirstlane_b32 {largestBits}, {largest}
            s_cmp_eq_u32 {largestBits}, {RowWriter.NegativeInfinity}
            s_cbranch_scc1 {prefix}topk_done
            s_add_u32 {kk}, {kk}, 1
            s_branch {prefix}topk
            """);
        w.Label(prefix + "topk_done");
        if (candidates)
        {
            // The candidates hold every token top-k keeps unless a block dropped a tied score at or above the cut-off.
            string ordered = w.S(), sign = w.S();
            w.L($"""
                s_cmp_lg_u32 {full}, 0
                s_cbranch_scc1 {prefix}complete
                s_cmp_eq_u32 {rowFlag}, 0
                s_cbranch_scc1 {prefix}complete
                v_readfirstlane_b32 {ordered}, {threshold}
                s_ashr_i32 {sign}, {ordered}, 31
                s_or_b32 {sign}, {sign}, 0x80000000
                s_xor_b32 {ordered}, {ordered}, {sign}
                s_cmp_gt_u32 {ordered}, {rowFlag}
                s_cbranch_scc1 {prefix}complete
                s_mov_b32 {full}, 1
                s_branch {prefix}source
                """);
            w.Label(prefix + "complete");
        }

        // Min-p: max(cut-off, max + ln(min-p)) (-∞ without min-p leaves it). Top-p: the bisection.
        string minimum = w.V(), total = w.V(), goal = w.V(), lo = w.V(), hi = w.V(), mid = w.V(), mass = w.V(), kept = w.V(), halvings = w.S(), within = w.SPair();
        w.L($"""
            s_cmp_lg_u32 {special}, 0
            s_cbranch_scc1 {prefix}cutoffs_done
            v_add_f32_e32 {minimum}, {a.Scalar("logminp")}, {max}
            v_max_f32_e32 {threshold}, {threshold}, {minimum}
            v_mov_b32_e32 {goal}, {a.Scalar("topp")}
            v_cmp_lt_f32_e32 vcc, 0, {goal}
            v_cmp_gt_f32_e64 {within}, 1.0, {goal}
            s_and_b64 vcc, vcc, {within}
            s_cbranch_vccz {prefix}cutoffs_done
            v_mov_b32_e32 {total}, 0
            """);
        w.Columns(cnt, Load, el => $"""
            {Score(xs[el.K])}
            {weight}
            v_add_f32_e32 {total}, {total}, {e}
            """);
        w.Reduce(RowWriter.Add, total);
        w.L($"""
            v_mul_f32_e32 {goal}, {a.Scalar("topp")}, {total}
            v_subrev_f32_e32 {lo}, 0x42200000, {max}
            v_max_f32_e32 {lo}, {lo}, {threshold}
            v_mov_b32_e32 {hi}, {max}
            s_mov_b32 {halvings}, 0
            """);
        w.Label(prefix + "halving");
        w.L($"""
            s_cmp_ge_u32 {halvings}, 24
            s_cbranch_scc1 {prefix}halved
            v_add_f32_e32 {mid}, {lo}, {hi}
            v_mul_f32_e32 {mid}, 0.5, {mid}
            v_mov_b32_e32 {mass}, 0
            """);
        w.Columns(cnt, Load, el => $"""
            {Score(xs[el.K])}
            {weight}
            v_cmp_le_f32_e32 vcc, {mid}, {sc}
            v_cndmask_b32_e32 {kept}, 0, {e}, vcc
            v_add_f32_e32 {mass}, {mass}, {kept}
            """);
        w.Reduce(RowWriter.Add, mass);
        w.L($"""
            v_cmp_ge_f32_e32 vcc, {mass}, {goal}
            v_cndmask_b32_e32 {lo}, {lo}, {mid}, vcc
            v_cndmask_b32_e32 {hi}, {mid}, {hi}, vcc
            s_add_u32 {halvings}, {halvings}, 1
            s_branch {prefix}halving
            """);
        w.Label(prefix + "halved");
        w.L($"v_mov_b32_e32 {threshold}, {lo}");
        w.Label(prefix + "cutoffs_done");

        // Σ weights; the target from the CPU's counter-based random value.
        string sum = w.V(), safe = w.V(), hash = w.S(), mixed = w.S(), target = w.V(), uniform = w.V();
        w.L($"v_mov_b32_e32 {sum}, 0");
        w.Columns(cnt, Load, el => $"""
            {Score(xs[el.K])}
            {weight}
            v_add_f32_e32 {sum}, {sum}, {e}
            """);
        w.Reduce(RowWriter.Add, sum);
        w.L($"""
            s_mul_i32 {mixed}, {row}, 0x85ebca6b
            s_xor_b32 {hash}, {streamBase}, {mixed}
            s_lshr_b32 {mixed}, {hash}, 16
            s_xor_b32 {hash}, {hash}, {mixed}
            s_mul_i32 {hash}, {hash}, 0x85ebca6b
            s_lshr_b32 {mixed}, {hash}, 13
            s_xor_b32 {hash}, {hash}, {mixed}
            s_mul_i32 {hash}, {hash}, 0xc2b2ae35
            s_lshr_b32 {mixed}, {hash}, 16
            s_xor_b32 {hash}, {hash}, {mixed}
            s_lshr_b32 {hash}, {hash}, 8
            v_cvt_f32_u32_e32 {uniform}, {hash}
            v_mul_f32_e32 {uniform}, 0x33800000, {uniform}
            v_mul_f32_e32 {target}, {uniform}, {sum}
            v_cmp_lt_f32_e32 vcc, 0, {sum}
            v_cndmask_b32_e32 {safe}, 1.0, {sum}, vcc
            """);

        // 3. The draw: each thread's contiguous chunk summed in order, four loads in flight.
        string chunk = w.S(), chunkEnd = w.S(), lastIndex = w.S(), step = w.S(), start = w.V(), end = w.V(), part = w.V();
        string[] js = [w.V(), w.V(), w.V(), w.V()], ats = [w.V(), w.V(), w.V(), w.V()], cxs = [w.V(), w.V(), w.V(), w.V()];
        w.L($"""
            s_add_u32 {chunk}, {cnt}, 0xff
            s_lshr_b32 {chunk}, {chunk}, 8
            s_sub_u32 {lastIndex}, {cnt}, 1
            v_mul_lo_u32 {start}, {chunk}, {w.Tid}
            v_min_u32_e32 {start}, {cnt}, {start}
            v_add_u32_e32 {end}, {chunk}, {start}
            v_min_u32_e32 {end}, {cnt}, {end}
            v_mov_b32_e32 {part}, 0
            s_mov_b32 {step}, 0
            """);
        string LoadFour() => $"""
            v_add_u32_e32 {js[0]}, {step}, {start}
            v_add_u32_e32 {js[1]}, 1, {js[0]}
            v_add_u32_e32 {js[2]}, 2, {js[0]}
            v_add_u32_e32 {js[3]}, 3, {js[0]}
            v_min_u32_e32 {ats[0]}, {lastIndex}, {js[0]}
            v_min_u32_e32 {ats[1]}, {lastIndex}, {js[1]}
            v_min_u32_e32 {ats[2]}, {lastIndex}, {js[2]}
            v_min_u32_e32 {ats[3]}, {lastIndex}, {js[3]}
            v_lshlrev_b32_e32 {ats[0]}, 2, {ats[0]}
            v_lshlrev_b32_e32 {ats[1]}, 2, {ats[1]}
            v_lshlrev_b32_e32 {ats[2]}, 2, {ats[2]}
            v_lshlrev_b32_e32 {ats[3]}, 2, {ats[3]}
            global_load_dword {cxs[0]}, {ats[0]}, {source}{coherent}
            global_load_dword {cxs[1]}, {ats[1]}, {source}{coherent}
            global_load_dword {cxs[2]}, {ats[2]}, {source}{coherent}
            global_load_dword {cxs[3]}, {ats[3]}, {source}{coherent}
            """;
        w.Label(prefix + "chunk");
        w.L($"""
            s_cmp_ge_u32 {step}, {chunk}
            s_cbranch_scc1 {prefix}chunk_end
            {LoadFour()}
            """);
        for (int u = 0; u < 4; u++)
        {
            w.L($"""
                {Score(cxs[u])}
                {weight}
                v_cmp_gt_u32_e32 vcc, {end}, {js[u]}
                v_cndmask_b32_e32 {e}, 0, {e}, vcc
                v_add_f32_e32 {part}, {part}, {e}
                """);
        }

        w.L($"""
            s_add_u32 {step}, {step}, 4
            s_branch {prefix}chunk
            """);
        w.Label(prefix + "chunk_end");

        // Thread 0 walks the chunks' sums in order: the chunk where the running total passes the target, and the total before it.
        string partAt = w.V(), lastChunk = w.V(), before = w.V(), running = w.V(), loaded = w.V(), scan = w.S(), scanAt = w.S(), saved = w.SPair(), minusOne = w.V();
        w.L($"""
            ds_write_b32 {tid4}, {part} offset:{extra}
            v_mov_b32_e32 {minusOne}, -1
            s_barrier
            v_cmp_eq_u32_e32 vcc, 0, {w.Tid}
            s_and_saveexec_b64 {saved}, vcc
            s_cbranch_execz {prefix}scan_done
            v_mov_b32_e32 {running}, 0
            v_mov_b32_e32 {before}, 0
            v_mov_b32_e32 {lastChunk}, -1
            s_mov_b32 {scan}, 0
            """);
        w.Label(prefix + "scan");
        w.L($"""
            s_cmp_ge_u32 {scan}, 0x{RowThreads:x}
            s_cbranch_scc1 {prefix}scanned
            s_lshl_b32 {scanAt}, {scan}, 2
            v_mov_b32_e32 {partAt}, {scanAt}
            ds_read_b32 {loaded}, {partAt} offset:{extra}
            v_cmp_lt_f32_e32 vcc, 0, {loaded}
            s_cbranch_vccz {prefix}scan_next
            v_mov_b32_e32 {lastChunk}, {scan}
            v_mov_b32_e32 {before}, {running}
            v_add_f32_e32 {running}, {running}, {loaded}
            v_cmp_lt_f32_e32 vcc, {target}, {running}
            s_cbranch_vccnz {prefix}scanned
            """);
        w.Label(prefix + "scan_next");
        w.L($"""
            s_add_u32 {scan}, {scan}, 1
            s_branch {prefix}scan
            """);
        w.Label(prefix + "scanned");
        w.L($"""
            ds_write_b32 {w.Zero}, {lastChunk} offset:{slotBase}
            ds_write_b32 {w.Zero}, {before} offset:{slotBase + 4}
            ds_write_b32 {w.Zero}, {minusOne} offset:{slotBase + 8}
            ds_write_b32 {w.Zero}, {w.Zero} offset:{slotBase + 12}
            """);
        w.Label(prefix + "scan_done");
        w.L($"""
            s_or_b64 exec, exec, {saved}
            s_barrier
            ds_read_b32 {lastChunk}, {w.Zero} offset:{slotBase}
            ds_read_b32 {before}, {w.Zero} offset:{slotBase + 4}
            """);

        // That chunk's thread walks it to the token where the running total passes the target (or its last weighted one).
        string chosen = w.V(), chosenWeight = w.V(), saved2 = w.SPair(), weighted = w.SPair();
        w.L($"""
            v_mov_b32_e32 {chosen}, -1
            v_mov_b32_e32 {chosenWeight}, 0
            v_cmp_eq_u32_e32 vcc, {lastChunk}, {w.Tid}
            s_and_saveexec_b64 {saved2}, vcc
            s_cbranch_execz {prefix}picked_none
            v_mov_b32_e32 {running}, {before}
            s_mov_b32 {step}, 0
            """);
        w.Label(prefix + "pick");
        w.L($"""
            s_cmp_ge_u32 {step}, {chunk}
            s_cbranch_scc1 {prefix}picked
            {LoadFour()}
            """);
        for (int u = 0; u < 4; u++)
        {
            w.L($"""
                {Score(cxs[u])}
                {weight}
                v_cmp_gt_u32_e32 vcc, {end}, {js[u]}
                v_cmp_lt_f32_e64 {weighted}, 0, {e}
                s_and_b64 vcc, vcc, {weighted}
                s_cbranch_vccz {prefix}pick_next{u}
                v_mov_b32_e32 {chosen}, {js[u]}
                v_mov_b32_e32 {chosenWeight}, {e}
                v_add_f32_e32 {running}, {running}, {e}
                v_cmp_lt_f32_e32 vcc, {target}, {running}
                s_cbranch_vccnz {prefix}picked
                """);
            w.Label($"{prefix}pick_next{u}");
        }

        w.L($"""
            s_add_u32 {step}, {step}, 4
            s_branch {prefix}pick
            """);
        w.Label(prefix + "picked");
        w.L($"""
            ds_write_b32 {w.Zero}, {chosen} offset:{slotBase + 8}
            ds_write_b32 {w.Zero}, {chosenWeight} offset:{slotBase + 12}
            """);
        w.Label(prefix + "picked_none");
        w.L($"""
            s_or_b64 exec, exec, {saved2}
            s_barrier
            ds_read_b32 {chosen}, {w.Zero} offset:{slotBase + 8}
            ds_read_b32 {chosenWeight}, {w.Zero} offset:{slotBase + 12}
            """);

        // A slot's token id (with candidates, read through candi; -1 stays -1).
        int maps = 0;
        string MapId(string index, string id)
        {
            string text = $"v_cvt_f32_i32_e32 {id}, {index}";
            if (!candidates)
            {
                return text;
            }

            string n = (maps++).ToString(CultureInfo.InvariantCulture), at = w.V(), value = w.V();
            return $"""
                {text}
                s_cmp_eq_u32 {mapped}, 0
                s_cbranch_scc1 {prefix}mapped{n}
                v_max_i32_e32 {at}, 0, {index}
                v_lshlrev_b32_e32 {at}, 2, {at}
                global_load_dword {value}, {at}, {candiRow}{coherent}
                v_cmp_lt_i32_e32 vcc, -1, {index}
                v_cndmask_b32_e32 {id}, {id}, {value}, vcc
                {prefix}mapped{n}:
                """;
        }

        // log2 of the probability in divQ into lg (scaled by 2^32 below the smallest normal, as LLVM's log).
        string lg = w.V(), scale = w.V(), back = w.V(), term = w.V(), entropy = w.V();
        string log2 = $"""
            v_cmp_gt_f32_e32 vcc, 0x00800000, {divQ}
            v_cndmask_b32_e32 {scale}, 1.0, {two32}, vcc
            v_mul_f32_e32 {lg}, {divQ}, {scale}
            v_log_f32_e32 {lg}, {lg}
            v_cndmask_b32_e32 {back}, 0, {thirtyTwo}, vcc
            v_sub_f32_e32 {lg}, {lg}, {back}
            """;

        // 4. The entropy over the weighted tokens.
        w.L($"""
            v_mov_b32_e32 {divD}, {safe}
            v_mov_b32_e32 {entropy}, 0
            """);
        w.Columns(cnt, Load, el => $"""
            {Score(xs[el.K])}
            {weight}
            v_mov_b32_e32 {divN}, {e}
            {divText}
            {log2}
            v_mul_f32_e32 {term}, {divQ}, {lg}
            v_cmp_lt_f32_e32 vcc, 0, {e}
            v_cndmask_b32_e32 {term}, 0, {term}, vcc
            v_sub_f32_e32 {entropy}, {entropy}, {term}
            """);
        w.Reduce(RowWriter.Add, entropy);

        // The token, its probability and the entropy (thread 0).
        string chosenId = w.V(), idValue = w.V(), probability = w.V(), saved3 = w.SPair();
        w.L($"""
            {MapId(chosen, chosenId)}
            v_max_f32_e32 {idValue}, 0, {chosenId}
            v_mov_b32_e32 {divN}, {chosenWeight}
            {divText}
            v_mov_b32_e32 {probability}, {divQ}
            v_cmp_eq_u32_e32 vcc, 0, {w.Tid}
            s_and_saveexec_b64 {saved3}, vcc
            global_store_dword {w.Zero}, {idValue}, {idsAt}
            global_store_dword {w.Zero}, {chosenId}, {statsRow}
            global_store_dword {w.Zero}, {probability}, {statsRow} offset:4
            global_store_dword {w.Zero}, {entropy}, {statsRow} offset:8
            s_or_b64 exec, exec, {saved3}
            """);

        // The five largest weights, each the largest not taken (the lower id on ties), with their probabilities.
        string[] taken = [w.S(), w.S(), w.S(), w.S()];
        string alternative = w.S(), statOffset = w.S(), pick = w.S(), bestWeight = w.V(), bestIndex = w.V(), larger = w.SPair(), other = w.SPair();
        string offsetValue = w.V(), altId = w.V(), saved4 = w.SPair();
        w.L($"""
            s_mov_b32 {taken[0]}, -1
            s_mov_b32 {taken[1]}, -1
            s_mov_b32 {taken[2]}, -1
            s_mov_b32 {taken[3]}, -1
            s_mov_b32 {alternative}, 0
            s_mov_b32 {statOffset}, 12
            """);
        w.Label(prefix + "top");
        w.L($"""
            s_cmp_ge_u32 {alternative}, 5
            s_cbranch_scc1 {prefix}top_end
            v_mov_b32_e32 {bestWeight}, 0
            v_mov_b32_e32 {bestIndex}, -1
            """);
        w.Columns(cnt, Load, el => $"""
            {Score(xs[el.K])}
            {weight}
            v_cmp_lt_f32_e64 {larger}, {bestWeight}, {e}
            v_cmp_ne_u32_e64 {other}, {taken[0]}, {el.Column}
            s_and_b64 {larger}, {larger}, {other}
            v_cmp_ne_u32_e64 {other}, {taken[1]}, {el.Column}
            s_and_b64 {larger}, {larger}, {other}
            v_cmp_ne_u32_e64 {other}, {taken[2]}, {el.Column}
            s_and_b64 {larger}, {larger}, {other}
            v_cmp_ne_u32_e64 {other}, {taken[3]}, {el.Column}
            s_and_b64 vcc, {larger}, {other}
            v_cndmask_b32_e32 {bestWeight}, {bestWeight}, {e}, vcc
            v_cndmask_b32_e32 {bestIndex}, {bestIndex}, {el.Column}, vcc
            """, columns: true);
        w.Reduce(RowWriter.ArgMax, bestWeight, bestIndex);
        w.L($"""
            v_readfirstlane_b32 {pick}, {bestIndex}
            {MapId(bestIndex, altId)}
            v_mov_b32_e32 {divN}, {bestWeight}
            {divText}
            v_mov_b32_e32 {offsetValue}, {statOffset}
            v_cmp_eq_u32_e32 vcc, 0, {w.Tid}
            s_and_saveexec_b64 {saved4}, vcc
            global_store_dword {offsetValue}, {altId}, {statsRow}
            global_store_dword {offsetValue}, {divQ}, {statsRow} offset:4
            s_or_b64 exec, exec, {saved4}
            s_mov_b32 {taken[3]}, {taken[2]}
            s_mov_b32 {taken[2]}, {taken[1]}
            s_mov_b32 {taken[1]}, {taken[0]}
            s_mov_b32 {taken[0]}, {pick}
            s_add_u32 {alternative}, {alternative}, 1
            s_add_u32 {statOffset}, {statOffset}, 8
            s_branch {prefix}top
            """);
        w.Label(prefix + "top_end");
        w.Label(prefix + "row_done");
    }

    // topk_candidates_f32 (CUDA's stage one of top-k sampling, topk ≤ 64): block b of row r (item g = r · bpr + b of the
    // grid-stride loop) takes scores [2048 · b, 2048 · b + 2048) of the row, 8 contiguous a thread, finds its k-th largest
    // distinct scaled score (k passes, each the largest below the last) and writes the scores at or above it with their
    // token ids, in vocabulary order (each thread's count of kept scores, then their exclusive prefix in a fixed-shape
    // scan: integers, exact in any order), to candv/candi[g · 64 …], the unused slots -∞ and -1. More than 64 at the
    // cut-off (ties): counted again keeping the scores strictly above it. flags[g] is the block's: 0 when complete; the
    // dropped cut-off's order-preserving bits when the retry dropped ties; 0xFFFFFFFF when the scores still overflow (the
    // sampler then walks the whole row). CUDA folds the flags with atomic maximums into one a row: here each block writes
    // its own and the sampler takes their maximum (rule 86). topk_greedy_f32 also writes parts[g · 3 …]: the block's best
    // scaled score (+∞ included, NaN never), how many of its scores equal it and the first of their ids, for
    // sample_candidates_greedy_f32 (CUDA's last block per row, its atomic counter and pending rows, as a step of the
    // sampling launch: still two launches).
    private static GcnKernelSource TopKCandidates(bool greedy)
    {
        var w = new RowWriter(greedy ? "topk_greedy_f32" : "topk_candidates_f32", arrays: 2, extraLds: 2 * 4 * RowThreads);
        var (pointerNames, scalarNames) = CandidateNames(greedy);
        var a = w.Arguments(pointerNames, scalarNames);
        var sampling = greedy ? SamplerSetup(w, a, SamplerForm.Greedy) : null;
        string vocab = a.Scalar("vocab"), blocks = a.Scalar("bpr"), topk = a.Scalar("topk"), invt = a.Scalar("invt");
        string byBlocks = AttentionReciprocal(w, blocks), lastIndex = w.S(), negInf = w.S(), tid8 = w.V(), tid4 = w.V(), minusOne = w.V(), negInfV = w.V(), posInfV = w.V();
        int scanA = w.ExtraLdsBase, scanB = w.ExtraLdsBase + 4 * RowThreads;
        w.L($"""
            s_sub_u32 {lastIndex}, {vocab}, 1
            s_mov_b32 {negInf}, {RowWriter.NegativeInfinity}
            v_lshlrev_b32_e32 {tid8}, 3, {w.Tid}
            v_lshlrev_b32_e32 {tid4}, 2, {w.Tid}
            v_mov_b32_e32 {minusOne}, -1.0
            v_mov_b32_e32 {negInfV}, {RowWriter.NegativeInfinity}
            v_mov_b32_e32 {posInfV}, {RowWriter.PositiveInfinity}
            """);
        w.BeginRows(a.Scalar("n"));
        var (row, block) = AttentionDivMod(w, w.Row, blocks, byBlocks);
        string first = w.S(), sliceFirst = w.S();
        w.L($"s_mul_i32 {first}, {row}, {a.Scalar("rowstride")}");
        w.L($"s_add_u32 {first}, {first}, {a.Scalar("rowoffset")}");
        string xb = w.ElementAddress(a.Pointer("logits"), first);
        w.L($"s_lshl_b32 {sliceFirst}, {block}, {(int)Math.Log2(CandidateSlice)}");
        string firstIndex = w.V();
        w.L($"v_add_u32_e32 {firstIndex}, {sliceFirst}, {tid8}           ; this thread's 8 scores");
        string[] js = new string[8], ats = new string[8], vs = new string[8];
        for (int i = 0; i < 8; i++)
        {
            (js[i], ats[i], vs[i]) = (w.V(), w.V(), w.V());
            w.L($"v_add_u32_e32 {js[i]}, {i}, {firstIndex}");
            w.L($"v_min_u32_e32 {ats[i]}, {lastIndex}, {js[i]}");
            w.L($"v_lshlrev_b32_e32 {ats[i]}, 2, {ats[i]}");
        }

        for (int i = 0; i < 8; i++)
        {
            w.L($"global_load_dword {vs[i]}, {ats[i]}, {xb}");
        }

        for (int i = 0; i < 8; i++)
        {
            w.L($"v_mul_f32_e32 {vs[i]}, {invt}, {vs[i]}");
            w.L($"v_cmp_gt_u32_e32 vcc, {vocab}, {js[i]}");
            w.L($"v_cndmask_b32_e32 {vs[i]}, {negInfV}, {vs[i]}, vcc       ; past the vocabulary: -∞");
        }

        if (greedy)
        {
            // The block's best score, its ties and the first of their ids.
            string best = w.V(), count = w.V(), firstId = w.V(), tie = w.V(), id = w.V();
            w.L($"v_mov_b32_e32 {best}, {RowWriter.NegativeInfinity}");
            for (int i = 0; i < 8; i++)
            {
                w.L($"v_max_f32_e32 {best}, {best}, {vs[i]}");
            }

            w.Reduce(RowWriter.Max, best);
            w.L($"v_mov_b32_e32 {count}, 0");
            w.L($"v_mov_b32_e32 {firstId}, {RowWriter.PositiveInfinity}");
            for (int i = 0; i < 8; i++)
            {
                w.L($"""
                    v_cmp_eq_f32_e32 vcc, {best}, {vs[i]}
                    v_cndmask_b32_e64 {tie}, 0, 1.0, vcc
                    v_add_f32_e32 {count}, {count}, {tie}
                    v_cvt_f32_u32_e32 {id}, {js[i]}
                    v_cndmask_b32_e32 {id}, {posInfV}, {id}, vcc
                    v_min_f32_e32 {firstId}, {firstId}, {id}
                    """);
            }

            w.Reduce(SumAndMin, count, firstId);
            string partsFirst = w.S();
            w.L($"s_mul_i32 {partsFirst}, {w.Row}, 3");
            string partsAt = w.ElementAddress(a.Pointer("parts"), partsFirst), savedParts = w.SPair();
            w.L($"""
                v_cmp_eq_u32_e32 vcc, 0, {w.Tid}
                s_and_saveexec_b64 {savedParts}, vcc
                global_store_dword {w.Zero}, {best}, {partsAt}
                global_store_dword {w.Zero}, {count}, {partsAt} offset:4
                global_store_dword {w.Zero}, {firstId}, {partsAt} offset:8
                s_or_b64 exec, exec, {savedParts}
                """);
        }

        // The k-th largest distinct score: k passes, each the largest below the last.
        string threshold = w.V(), largest = w.V(), below = w.SPair(), kk = w.S(), largestBits = w.S();
        w.L($"""
            v_mov_b32_e32 {threshold}, {RowWriter.PositiveInfinity}
            s_mov_b32 {kk}, 0
            """);
        w.Label(".Ltopk");
        w.L($"""
            s_cmp_ge_u32 {kk}, {topk}
            s_cbranch_scc1 .Ltopk_done
            v_mov_b32_e32 {largest}, {RowWriter.NegativeInfinity}
            """);
        for (int i = 0; i < 8; i++)
        {
            w.L($"""
                v_cmp_gt_f32_e32 vcc, {threshold}, {vs[i]}
                v_cmp_lt_f32_e64 {below}, {largest}, {vs[i]}
                s_and_b64 vcc, vcc, {below}
                v_cndmask_b32_e32 {largest}, {largest}, {vs[i]}, vcc
                """);
        }

        w.Reduce(RowWriter.Max, largest);
        w.L($"""
            v_mov_b32_e32 {threshold}, {largest}
            v_readfirstlane_b32 {largestBits}, {largest}
            s_cmp_eq_u32 {largestBits}, {RowWriter.NegativeInfinity}
            s_cbranch_scc1 .Ltopk_done
            s_add_u32 {kk}, {kk}, 1
            s_branch .Ltopk
            """);
        w.Label(".Ltopk_done");

        // The kept scores: at or above the cut-off (strictly above it on the retry), above -∞; their count, its prefix.
        string strict = w.S(), atOrAbove = w.SPair(), count2 = w.V(), sum = w.V(), other = w.V(), source = w.V(), total = w.S(), totalValue = w.V(), oneInt = w.V();
        string[] keeps = new string[8];
        for (int i = 0; i < 8; i++)
        {
            keeps[i] = w.SPair();
        }

        w.L($"s_mov_b64 {atOrAbove}, -1");
        w.L($"s_mov_b32 {strict}, 0");
        w.Label(".Lcount");
        w.L($"v_mov_b32_e32 {count2}, 0");
        for (int i = 0; i < 8; i++)
        {
            w.L($"""
                v_cmp_le_f32_e64 {keeps[i]}, {threshold}, {vs[i]}
                s_and_b64 {keeps[i]}, {keeps[i]}, {atOrAbove}
                v_cmp_lt_f32_e64 {below}, {threshold}, {vs[i]}
                s_or_b64 {keeps[i]}, {keeps[i]}, {below}
                v_cmp_lt_f32_e64 {below}, {negInf}, {vs[i]}
                s_and_b64 {keeps[i]}, {keeps[i]}, {below}
                v_cndmask_b32_e64 {oneInt}, 0, 1, {keeps[i]}
                v_add_u32_e32 {count2}, {count2}, {oneInt}
                """);
        }

        // Inclusive prefix over the block's threads in two LDS buffers (a fixed shape, a barrier a step).
        w.L($"""
            v_mov_b32_e32 {sum}, {count2}
            ds_write_b32 {tid4}, {sum} offset:{scanA}
            s_barrier
            """);
        int from = scanA, to = scanB;
        for (int distance = 1; distance < RowThreads; distance *= 2)
        {
            w.L($"""
                v_subrev_u32_e32 {source}, {4 * distance}, {tid4}
                v_cmp_le_u32_e32 vcc, {distance}, {w.Tid}
                v_cndmask_b32_e32 {source}, {tid4}, {source}, vcc
                ds_read_b32 {other}, {source} offset:{from}
                v_cndmask_b32_e32 {other}, 0, {other}, vcc
                v_add_u32_e32 {sum}, {sum}, {other}
                ds_write_b32 {tid4}, {sum} offset:{to}
                s_barrier
                """);
            (from, to) = (to, from);
        }

        w.L($"""
            ds_read_b32 {totalValue}, {w.Zero} offset:{from + 4 * (RowThreads - 1)}
            v_readfirstlane_b32 {total}, {totalValue}
            s_barrier
            s_cmp_le_u32 {total}, {CandidateSlots}
            s_cbranch_scc1 .Lcounted
            s_cmp_lg_u32 {strict}, 0
            s_cbranch_scc1 .Lcounted
            s_mov_b32 {strict}, 1
            s_mov_b64 {atOrAbove}, 0
            s_branch .Lcount
            """);
        w.Label(".Lcounted");

        // The kept scores in order into the block's slots (the first 64), then the unused slots, then the block's flag.
        string slot = w.V(), slotAt = w.V(), id2 = w.V(), room = w.SPair(), saved = w.SPair(), slots64 = w.S();
        w.L($"s_movk_i32 {slots64}, 0x{CandidateSlots:x}");
        int candidatesAt = w.Product64(w.Row, slots64);
        string candvRow = w.Base(a.Pointer("candv"), candidatesAt), candiRow = w.Base(a.Pointer("candi"), candidatesAt);
        w.L($"v_sub_u32_e32 {slot}, {sum}, {count2}                   ; the exclusive prefix: this thread's first slot");
        for (int i = 0; i < 8; i++)
        {
            w.L($"""
                v_cmp_gt_u32_e64 {room}, 64, {slot}
                s_and_b64 {room}, {room}, {keeps[i]}
                s_and_saveexec_b64 {saved}, {room}
                v_lshlrev_b32_e32 {slotAt}, 2, {slot}
                v_cvt_f32_u32_e32 {id2}, {js[i]}
                global_store_dword {slotAt}, {vs[i]}, {candvRow}
                global_store_dword {slotAt}, {id2}, {candiRow}
                s_or_b64 exec, exec, {saved}
                v_cndmask_b32_e64 {oneInt}, 0, 1, {keeps[i]}
                v_add_u32_e32 {slot}, {slot}, {oneInt}
                """);
        }

        string flag = w.S(), ordered = w.S(), sign = w.S(), flagValue = w.V(), unused = w.SPair();
        w.L($"""
            v_cmp_gt_u32_e32 vcc, 64, {w.Tid}
            v_cmp_le_u32_e64 {unused}, {total}, {w.Tid}
            s_and_b64 vcc, vcc, {unused}
            s_and_saveexec_b64 {saved}, vcc
            global_store_dword {tid4}, {negInfV}, {candvRow}
            global_store_dword {tid4}, {minusOne}, {candiRow}
            s_or_b64 exec, exec, {saved}
            v_readfirstlane_b32 {ordered}, {threshold}
            s_ashr_i32 {sign}, {ordered}, 31
            s_or_b32 {sign}, {sign}, 0x80000000
            s_xor_b32 {ordered}, {ordered}, {sign}
            s_cmp_eq_u32 {strict}, 0
            s_cselect_b32 {flag}, 0, {ordered}
            s_cmp_gt_u32 {total}, {CandidateSlots}
            s_cselect_b32 {flag}, -1, {flag}
            v_mov_b32_e32 {flagValue}, {flag}
            """);
        w.StoreFirstLane((w.ElementAddress(a.Pointer("flags"), w.Row), flagValue));
        if (sampling is not null)
        {
            // With counters (not 0): the row's last block samples it (sample_candidates_greedy_f32's body over the blocks'
            // candidates, flags and parts, read with glc), so a greedy row takes one launch.
            string counters = a.Pointer("counters");
            w.L($"""
                s_cmp_eq_u64 {counters}, 0
                s_cbranch_scc1 .Lrow_next
                """);
            BlockArrive(w, counters, row, blocks, ".Lrow_next", scanA);
            SamplerRow(w, a, SamplerForm.Greedy, sampling, row, ".Ls_", glc: true);
            w.Label(".Lrow_next");
        }

        w.EndRows();
        return w.Source();
    }
}
