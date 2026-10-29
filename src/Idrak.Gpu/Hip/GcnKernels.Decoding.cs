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
//   k splits are the plain products' (HipBackend.GemvSplitCount) and their sums are added in split order by a second
//   launch, never by CUDA's atomic arrival counter (rule 86): int8_gemv_finish_f32 for the gated products,
//   gemv_multi_finish_f32 for the multi-output ones (which also writes act(gate) · up for a gate/up pair), and for the
//   product followed by the residual addition and the next norm, gemv_addnorm_f32 (CUDA's *_gemv_addnorm_f32 and
//   *_gemv_normaddnorm_f32 do this in the product's last block: here the plain product, then this row kernel).
// - attention_rows_f32 and attention_rows_bf16: decoding over a float32 or bfloat16 cache, each row from its sequence's
//   start (GcnKernels.Attention.cs's kernel with a table of starts); segment_spans_f32 writes the key range of each row
//   of packed sequences (causal, from the sequence's start, a window), so attention_spans_f32 runs them.
// - The rows of decoding: argmax_f32, class_match_f32, add_rms_norm_affine_f32, rms_norm_rope_f32 (a block a row, the
//   reductions in RowWriter's fixed tree), rope_f32, one_hot_f32 and history_push_f32 (an element a work-item,
//   grid-stride), penalize_rows_f32 (a block a row: the row copied, then each distinct id of the window penalized once).
//
// For the profile's labels and --bench-hip's rates (HipBackend.Cost, HipBackend.ProductCost): each kernel, its
// arguments (pointers, then 32-bit values, float ones as their bits; PackArguments packs them), its launch, and its
// floating-point operations and global-memory bytes per unit (a division, exp or rsq counted as the instructions do):
//
// | Kernel                    | Arguments                                                                               | Launch (block)                   | Unit            | Flops | Bytes |
// |---------------------------|-----------------------------------------------------------------------------------------|----------------------------------|-----------------|-------|-------|
// | *_gemv_silu/gelu[_rR]_f32 | the packed product's (x is the gate), then up                                           | ⌈w/64⌉ × splits (64)             | as *_gemv_f32 (ProductCost)     |
// | *_gemv_multi[_rR]_f32     | x, part, m, k, chunk, splits, counters, blocks, nmax; 3 × (q, s, y, bias, n)            | ⌈w/64⌉ × splits × products (64)  | as *_gemv_f32 (ProductCost)     |
// | gemv_multi_finish_f32     | part, hidden, splits, m, nmax, act; 3 × (s, y, bias, n)                                 | ⌈m·n/256⌉ or fewer × products     | output          | 1 + splits | 4·(splits + 2) |
// | gemv_addnorm_f32          | part, s, y, residual, sum, gain, normalized, ygain; cols, splits, size, eps, offset, yeps, yoffset, flags, n | rows (256) | element | 8 + splits | 4·(splits + 8) |
// | attention_rows_f32        | attention_decode_f32's, then starts; hpr before n                                       | rows x splits (256)              | row × position × dim | 4 | 8 |
// | attention_rows_bf16       | attention_decode_bf16's, then starts; hpr before n                                      | rows x splits (256)              | row × position × dim | 4 | 4 |
// | segment_spans_f32         | starts, begins, ends; rows, steps, window, n                                            | ⌈n/256⌉ or fewer (256)           | row             | 0     | 12    |
// | argmax_f32                | x, y; cols, n                                                                           | rows (256)                       | element         | 1     | 4     |
// | class_match_f32           | predictions, targets, y; cols, threshold, n                                             | rows (256)                       | element         | 2     | 8     |
// | add_rms_norm_affine_f32   | a, b, sum, gain, y; cols, eps, offset, n                                                | rows (256)                       | element         | 6     | 24    |
// | rms_norm_rope_f32         | x, gain, cos, sin, positions, y; cols, eps, offset, heads, steps, half, interleaved, n   | rows (256)                       | element         | 7     | 20    |
// | rope_f32                  | x, y, cos, sin, positions; heads, steps, dim, half, interleaved, sign, n                | ⌈n/256⌉ or fewer (256)           | pair            | 7     | 24    |
// | one_hot_f32               | indices, y; classes, last, n                                                            | ⌈n/256⌉ or fewer (256)           | element         | 0     | 8     |
// | history_push_f32          | ids, history, length; capacity, rows, n                                                 | ⌈n/256⌉ or fewer (256)           | row             | 0     | 8     |
// | penalize_rows_f32         | logits, work, history, length; vocabulary, stride, offset, capacity, lastn, repeat, presence, frequency, rows, n | rows (256) | element | 0 | 8 |
//
// (w: the widest product's packed words a row; R: the rows a kernel is written for. penalize_rows_f32 counts the row's
// copy; its few penalized ids, each over the window, are left out. gemv_multi_finish_f32's act: 0 none, else the gated
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
    }

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
            foreach (string variant in (string[])["silu", "gelu", "multi"])
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
        return FusedGemvNames[(f * 3 + (int)variant - 1) * 4 + i];
    }

    // ---- Arguments and packing -------------------------------------------------------------------------------------

    /// <summary>A gated product's arguments: the packed product's (<see cref="PackedGemvArguments"/>, x being the gate), then up at 72.</summary>
    internal static IReadOnlyList<GcnArgument> GatedGemvArguments => _gatedGemvArguments ??= [.. PackedGemvArguments, GcnArgument.Buffer(72, "up")];

    private static GcnArgument[]? _gatedGemvArguments;

    /// <summary>Bytes of a gated product's arguments.</summary>
    internal const int GatedGemvArgumentBytes = 80;

    /// <summary>Bytes of a multi-output product's shared arguments (x, part; m, k, chunk, splits; counters; blocks, nmax).</summary>
    internal const int MultiGemvSharedBytes = 48;

    /// <summary>Bytes of each product's arguments in a multi-output product's (q, s, y, bias, n and the padding to 8).</summary>
    internal const int MultiGemvProductBytes = 40;

    /// <summary>The most products a multi-output product takes (grid z).</summary>
    internal const int MultiGemvProducts = 3;

    /// <summary>
    /// A multi-output product's arguments, CUDA's (RunMany's): x, part (0: each output written whole), m, k, chunk,
    /// splits, counters (unused), blocks, nmax (the widest product's columns: part's row length), then for products 0
    /// to 2 the packed weights, scales (0: none), output, bias (0: none) and columns, from 48 + 40j.
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

        return [.. list];
    }

    /// <summary>Bytes of a multi-output product's arguments (the last product's width ends them, with no padding after it).</summary>
    internal const int MultiGemvArgumentBytes = MultiGemvSharedBytes + MultiGemvProductBytes * (MultiGemvProducts - 1) + 36;

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
        ReadOnlySpan<MultiProduct> products)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(arguments.Length, MultiGemvArgumentBytes);
        Span<ulong> values = stackalloc ulong[9 + 5 * MultiGemvProducts];
        (values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8]) = (x, part, m, k, chunk, splits, 0UL, blocks, nmax);
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
            foreach (var variant in (PackedGemvVariant[])[PackedGemvVariant.Silu, PackedGemvVariant.Gelu, PackedGemvVariant.Multi])
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
        kernels.Add(RmsNormRope());
        kernels.Add(Rope());
        kernels.Add(OneHot());
        kernels.Add(HistoryPush());
        kernels.Add(PenalizeRows());
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

    // Each row (a head's vector) RMS-normalized with gain, then rotated as Rope with sign 1 at the position of its step
    // (positions[(row / heads) mod steps]); the dimensions past 2 · half normalized alone (CpuBackend.RmsNormRope). A
    // thread rotates the pairs p = tid + 256 i, each read and written by it alone, so y may be x.
    private static GcnKernelSource RmsNormRope()
    {
        var w = new RowWriter("rms_norm_rope_f32", arrays: 1);
        var a = w.Arguments(["x", "gain", "cos", "sin", "positions", "y"], ["cols", "eps", "offset", "heads", "steps", "half", "interleaved", "n"]);
        string cols = a.Scalar("cols"), half = a.Scalar("half"), heads = a.Scalar("heads"), steps = a.Scalar("steps"), offsetValue = a.Scalar("offset");
        string gain = a.Pointer("gain");
        string rc = w.ReciprocalOf(cols), byHeads = AttentionReciprocal(w, heads), bySteps = AttentionReciprocal(w, steps);
        string shift = w.S(), pairStep = w.S(), rest = w.S();
        w.L($"s_cmp_lg_u32 {a.Scalar("interleaved")}, 0");
        w.L($"s_cselect_b32 {shift}, 1, 0                              ; interleaved: pair p at 2p and 2p + 1, else p and p + half");
        w.L($"s_cselect_b32 {pairStep}, 1, {half}");
        w.L($"s_lshl_b32 {rest}, {half}, 1");
        w.BeginRows(a.Scalar("n"));
        int offset = w.RowOffset(cols);
        string xb = w.Base(a.Pointer("x"), offset), yb = w.Base(a.Pointer("y"), offset), acc = w.V();
        var xs = w.Vs();
        w.L($"v_mov_b32_e32 {acc}, 0");
        w.Columns(cols, e => $"global_load_dword {xs[e.K]}, {e.Off}, {xb}{e.Imm}", e => $"v_fma_f32 {acc}, {xs[e.K]}, {xs[e.K]}, {acc}");
        w.Reduce(RowWriter.Add, acc);
        string r = w.V();
        w.L($"v_mul_f32_e32 {r}, {acc}, {rc}");
        w.L($"v_add_f32_e32 {r}, {a.Scalar("eps")}, {r}");
        w.L($"v_rsq_f32_e32 {r}, {r}");

        // The row's position and its rows of the tables.
        var (step, _) = AttentionDivMod(w, w.Row, heads, byHeads);
        var (_, at) = AttentionDivMod(w, step, steps, bySteps);
        string positionAt = w.ElementAddress(a.Pointer("positions"), at), loaded = w.V(), position = w.S(), table = w.S();
        w.L($"global_load_dword {loaded}, {w.Zero}, {positionAt}");
        w.L($"v_cvt_i32_f32_e32 {loaded}, {loaded}");
        w.L($"v_readfirstlane_b32 {position}, {loaded}");
        w.L($"s_mul_i32 {table}, {position}, {half}");
        string cb = w.ElementAddress(a.Pointer("cos"), table), sb = w.ElementAddress(a.Pointer("sin"), table);

        // The pairs.
        string first = w.S(), pair = w.V(), i = w.V(), j = w.V(), pairAt = w.V(), saved = w.SPair();
        string xi = w.V(), xj = w.V(), gi = w.V(), gj = w.V(), c = w.V(), s = w.V(), t = w.V(), yi = w.V(), yj = w.V();
        w.L($"s_mov_b32 {first}, 0");
        w.Label(".Lpairs");
        w.L($"""
            s_cmp_ge_u32 {first}, {half}
            s_cbranch_scc1 .Lpairs_end
            v_add_u32_e32 {pair}, {first}, {w.Tid}
            v_cmp_gt_u32_e32 vcc, {half}, {pair}
            s_and_saveexec_b64 {saved}, vcc
            v_lshlrev_b32_e32 {i}, {shift}, {pair}
            v_add_u32_e32 {j}, {pairStep}, {i}
            v_lshlrev_b32_e32 {i}, 2, {i}
            v_lshlrev_b32_e32 {j}, 2, {j}
            v_lshlrev_b32_e32 {pairAt}, 2, {pair}
            global_load_dword {xi}, {i}, {xb}
            global_load_dword {xj}, {j}, {xb}
            global_load_dword {gi}, {i}, {gain}
            global_load_dword {gj}, {j}, {gain}
            global_load_dword {c}, {pairAt}, {cb}
            global_load_dword {s}, {pairAt}, {sb}
            v_add_f32_e32 {gi}, {offsetValue}, {gi}
            v_add_f32_e32 {gj}, {offsetValue}, {gj}
            v_mul_f32_e32 {xi}, {xi}, {r}
            v_mul_f32_e32 {xi}, {xi}, {gi}
            v_mul_f32_e32 {xj}, {xj}, {r}
            v_mul_f32_e32 {xj}, {xj}, {gj}
            v_mul_f32_e32 {t}, {xj}, {s}
            v_fma_f32 {yi}, {xi}, {c}, -{t}
            v_mul_f32_e32 {t}, {xi}, {s}
            v_fma_f32 {yj}, {xj}, {c}, {t}
            global_store_dword {i}, {yi}, {yb}
            global_store_dword {j}, {yj}, {yb}
            s_or_b64 exec, exec, {saved}
            s_add_u32 {first}, {first}, 0x{RowThreads:x}
            s_branch .Lpairs
            """);
        w.Label(".Lpairs_end");

        // The dimensions past 2 · half: normalized alone.
        string column = w.V(), columnAt = w.V(), value = w.V(), g = w.V();
        w.L($"s_mov_b32 {first}, {rest}");
        w.Label(".Lrest");
        w.L($"""
            s_cmp_ge_u32 {first}, {cols}
            s_cbranch_scc1 .Lrest_end
            v_add_u32_e32 {column}, {first}, {w.Tid}
            v_cmp_gt_u32_e32 vcc, {cols}, {column}
            s_and_saveexec_b64 {saved}, vcc
            v_lshlrev_b32_e32 {columnAt}, 2, {column}
            global_load_dword {value}, {columnAt}, {xb}
            global_load_dword {g}, {columnAt}, {gain}
            v_add_f32_e32 {g}, {offsetValue}, {g}
            v_mul_f32_e32 {value}, {value}, {r}
            v_mul_f32_e32 {value}, {value}, {g}
            global_store_dword {columnAt}, {value}, {yb}
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
}
