// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;

namespace Idrak.Gpu.Hip;

// HIP's training kernels in GCN (plan 35, G5b), with CUDA's names and arguments where CUDA has the kernel
// (PtxKernels.cs, .Advanced.cs, .Pointwise.cs, .Decoder.cs, .Rows.cs, .SpansBackward.cs) and the CPU backend's results
// (CpuBackend*.cs) within the device-against-CPU tolerance: the backward passes (unary, softmax, RMS norm, grouped norms,
// gated activations, extremes, selection, clamps, powers, leaky ReLU), the optimizers (SGD with momentum, Adam, 8-bit Adam,
// AdamW over many tensors with its clipping norm, the clipping factor, dropout), the reductions (sums of rows, columns,
// axes, groups and squares, the whole sum, scatter-add, broadcasts, weight norm) and the attention backward passes
// (GcnKernels.TrainingAttention below; tiled over a cache and over key ranges).
//
// Three shapes of kernel:
// - Element-wise (TrainingElementwise): the frame of GcnKernels.Pointwise.cs (blocks of 256, a dwordx4 main loop over
//   whole chunks of 256 elements a wave while every buffer moved four at a time is 16-byte aligned, a grid-stride tail of
//   one element a work-item), generalized to several written buffers, loads and stores gated by a flag bit (an optional
//   gradient is never touched), the element's index (dropout's hash) and its group (i / inner mod groups, by LLVM's
//   unsigned division through a reciprocal made once). "Single" kernels (sum_axis_f32, scatter_walk_f32,
//   clip_factor_f32) take the tail alone: one element a work-item, grid-stride. Every product, quotient and sum is the
//   CPU's, in the CPU's order (its vector path where it has one): exact where the CPU's arithmetic is.
// - Rows (RowWriter, GcnKernels.Rows.cs): one block of 256 a row, grid-stride, the block's sums in RowWriter's fixed LDS
//   tree; the whole sum, the sums of squares and the chunks of the split group reductions write one value a block that a
//   second launch adds in block (chunk) order: never atomics, the same bits on every grid and device (rule 86).
// - Columns (sum_rows_f32, sum_cols_strided_f32): a thread a column, rows in order; chunks of rows in grid y store
//   partials that fold_chunks_f32 (GcnKernels.Rows.cs) adds in chunk order.
//
// Scatter-add (an embedding's gradient) is CUDA's, which is deterministic without atomics: scatter_chain_u32 links each
// position to the next with the same id (the LDS holds a tile of ids at a time, each lane compares its own against them
// and stops once it found the next), marking the first of each id; scatter_walk_f32 has one work-item per (head, column)
// add the id's rows in position order into the table, so every element of the table is written once, in the CPU's order.
//
// For --profile's labels and --bench-hip's rates (HipBackend.TrainingCost): each kernel, its arguments (pointers, then
// 32-bit values, the float ones as their bits; PackArguments packs them), its launch, and its floating-point operations
// and global-memory bytes per unit (an exp, log, rsq, rcp or a division step counted as one operation; integer work and
// selects not counted):
//
// | Kernel                 | Arguments                                                                   | Launch (blocks of 256)              | Unit                  | Flops | Bytes |
// |------------------------|-----------------------------------------------------------------------------|-------------------------------------|-----------------------|-------|-------|
// | sigmoid_bwd_f32        | x, y, dy, dx; n                                                             | element-wise                        | element               | 3     | 16    |
// | tanh_bwd_f32           | x, y, dy, dx; n                                                             | element-wise                        | element               | 3     | 16    |
// | relu_bwd_f32           | x, y, dy, dx; n                                                             | element-wise                        | element               | 1     | 16    |
// | square_bwd_f32         | x, y, dy, dx; n                                                             | element-wise                        | element               | 3     | 16    |
// | abs_bwd_f32            | x, y, dy, dx; n                                                             | element-wise                        | element               | 2     | 16    |
// | exp_bwd_f32            | x, y, dy, dx; n                                                             | element-wise                        | element               | 2     | 16    |
// | log_bwd_f32            | x, y, dy, dx; n                                                             | element-wise                        | element               | 12    | 16    |
// | gelu_bwd_f32           | x, y, dy, dx; n                                                             | element-wise                        | element               | 52    | 16    |
// | gelu_erf_bwd_f32       | x, y, dy, dx; n                                                             | element-wise                        | element               | 44    | 16    |
// | sqrt_bwd_f32           | x, y, dy, dx; n                                                             | element-wise                        | element               | 13    | 16    |
// | sin_bwd_f32            | x, y, dy, dx; n                                                             | element-wise                        | element               | 34    | 16    |
// | cos_bwd_f32            | x, y, dy, dx; n                                                             | element-wise                        | element               | 34    | 16    |
// | silu_bwd_f32           | x, y, dy, dx; n                                                             | element-wise                        | element               | 34    | 16    |
// | extremum_bwd_f32       | a, b, dy, da, db; flags, n                                                  | element-wise                        | element               | 2     | 28    |
// | pow_bwd_f32            | x, dy, dx; e, p, flags, n                                                   | element-wise                        | element               | 10    | 16    |
// | clamp_bwd_f32          | x, dy, dx; lo, hi, n                                                        | element-wise                        | element               | 1     | 16    |
// | leaky_relu_bwd_f32     | x, dy, dx; slope, n                                                         | element-wise                        | element               | 2     | 16    |
// | where_bwd_f32          | condition, dy, da, db; flags, n                                             | element-wise                        | element               | 2     | 24    |
// | gated_act_bwd_f32      | gate, up, dy, dgate, dup; kind, flags, n                                    | element-wise                        | element               | 40    | 28    |
// | gated_act_bwd_bf16_f32 | pg, pu, dy, dgate, dup; kind, flags, count, n (n = ⌈count / 2⌉ words)        | single, a work-item a word          | element               | 40    | 24    |
// | softmax_bwd_f32        | y, dy, dx; cols, log, n                                                     | rows (or fewer)                     | element               | 5     | 20    |
// | rms_norm_backward_f32  | dy, y, inv, dx; cols, n                                                     | rows (or fewer)                     | element               | 6     | 24    |
// | norm_bwd_f32           | dxhat, xhat, s1, s2, invstd, dx; groups, inner, m, n                        | element-wise                        | element               | 30    | 28    |
// | sgd_momentum_f32       | p, g, v; nlr, momentum, n                                                   | element-wise                        | element               | 4     | 20    |
// | adam_f32               | p, g, m, v; lr, b1, b2, c1, c2, eps, n                                      | element-wise                        | element               | 28    | 28    |
// | adam8_f32              | p, g, m, v, absmax, map; lr, b1, b2, c1, c2, eps, count, gscale, decay, n   | blocks of 256 elements (or fewer)   | element               | 50    | 14    |
// | multi_sumsq_f32        | chunks, table, partials; n                                                  | chunks of 4,096 (or fewer)          | element               | 2     | 4     |
// | multi_adamw_f32        | chunks, table, factor; lr, decay, b1, b2, c1, c2, eps, zero, n              | chunks of 4,096 (or fewer)          | element               | 30    | 32    |
// | clip_factor_f32        | x, y; maxnorm, n                                                            | single (n = 1)                      | value                 | 20    | 8     |
// | dropout_f32            | x, y; p, scale, seed, accumulate, n                                         | element-wise                        | element               | 2     | 8     |
// | add_dropout_f32        | r, x, y; p, scale, seed, n                                                  | element-wise                        | element               | 2     | 12    |
// | sum_rows_f32           | x, y; rows, cols                                                            | ceil(cols/256) (or fewer)           | element of x          | 1     | 4     |
// | sum_cols_strided_f32   | x, y, partials; rows, cols, ld, chunk                                       | ceil(cols/256) (or fewer) x chunks  | element of x          | 1     | 4     |
// | sum_f32                | x, out, partials; n, scale                                                  | min(ceil(n/256), 1024)              | element               | 1     | 4     |
// | sum_blocks_f32         | partials, out; count, scale, accumulate, n                                  | 1                                   | partial               | 1     | 4     |
// | sumsq_f32              | x, total, partials; count, direct, n                                        | n = min(ceil(count/256), 1024)      | element               | 2     | 4     |
// | sum_axis_f32           | x, y; dim, inner, scale, accumulate, n                                      | single, n = outer x inner           | element of x          | 1     | 4     |
// | sum_axis_rows_f32      | x, y; dim, scale, accumulate, n                                             | rows (or fewer), n = outer          | element of x          | 1     | 4     |
// | broadcast_axis_f32     | dy, dx; diminner, inner, scale, n                                           | element-wise over dx                | element of dx         | 2     | 12    |
// | scatter_chain_u32      | indices, chain; count, maxindex, n                                          | ceil(count/256) (or fewer)          | position x position   | 0     | 0.02  |
// | scatter_walk_f32       | dy, chain, indices, dtable; dim, maxindex, count, n                         | single, n = count x dim             | element of dy         | 1     | 12    |
// | norm_stats_f32         | x, mean, var, invstd; outer, groups, inner, eps                             | groups (or fewer)                   | element               | 3     | 8     |
// | norm_stats_part_f32    | x, part; outer, groups, inner, chunks                                       | groups x chunks (or fewer)          | element               | 3     | 8     |
// | norm_stats_final_f32   | part, mean, var, invstd; outer, groups, inner, chunks, eps                  | groups (or fewer)                   | chunk                 | 10    | 8     |
// | norm_apply_f32         | x, mean, invstd, y; groups, inner, n                                        | element-wise                        | element               | 3     | 16    |
// | group_reduce_f32       | a, b, suma, sumab; outer, groups, inner, hasb                               | groups (or fewer)                   | element               | 3     | 8     |
// | group_reduce_part_f32  | a, b, part; outer, groups, inner, chunks, hasb                              | groups x chunks (or fewer)          | element               | 3     | 8     |
// | group_reduce_final_f32 | part, suma, sumab; groups, chunks, hasb                                     | groups (or fewer)                   | chunk                 | 2     | 8     |
// | group_scale_shift_f32  | x, scale, shift, y; groups, inner, flags, n                                 | element-wise                        | element               | 2     | 16    |
// | weight_norm_f32        | gain, direction, weight; cols, n                                            | rows (or fewer)                     | element               | 3     | 12    |
// | weight_norm_bwd_f32    | gain, direction, dweight, dgain, ddirection; cols, flags, n                 | rows (or fewer)                     | element               | 8     | 24    |
// | attn_bwd_d_f32         | o, dout, dd; dim, n                                                         | rows (or fewer)                     | element               | 2     | 8     |
// | attn_bwd_q_f32         | q, keys, values, dout, lse, delta, dq, dk, dv; rph, steps, cap, dim, scale, window, softcap, n | rows (or fewer) | row x position x dim | 6 | 8 |
// | attn_bwd_kv_f32        | (as attn_bwd_q_f32), n = heads x cap                                        | key positions (or fewer)            | row x position x dim  | 8     | 8     |
// | attn_spans_bwd_q_f32   | q, keys, values, starts, ends, dout, lse, delta, dq, dk, dv; rows, keyrows, dim, scale, group, hpt, softcap, heads, n | rows (or fewer) | row x key x dim | 6 | 8 |
// | attn_spans_bwd_kv_f32  | (as attn_spans_bwd_q_f32), n = kv heads x keyrows                           | key positions (or fewer)            | row x key x dim       | 8     | 8     |
//
// "element-wise" launches as GcnKernels.Pointwise's (blocks of 256, about one per 1,024 elements, within the grid and
// 2^29 work-items: HipBackend.PointwiseBlocks); "or fewer": the kernel walks its rows, blocks or positions grid-stride.
// The counts stay below 2^30 (byte offsets are 32-bit). Optional storages are passed as another valid pointer with
// their flag bit clear, as CUDA passes them; the kernels never touch them.
internal static partial class GcnKernels
{
    /// <summary>Elements a quantization block of 8-bit Adam holds (EightBitMoments.BlockSize): one a thread of the block.</summary>
    internal const int Adam8Block = 256;

    /// <summary>Elements one chunk of the many-tensor AdamW takes (CUDA's 4,096): a block a chunk.</summary>
    internal const int MultiAdamChunk = 4096;

    /// <summary>Elements a split group reduction's block takes (CUDA's GroupChunk): 32 a thread of the block.</summary>
    internal const int GroupChunk = 32 * RowThreads;

    /// <summary>The flag bits of pow_bwd_f32's exponent (CUDA's PowFlags, the tests MathF.Pow makes on it).</summary>
    internal const uint PowGradientOdd = 1, PowGradientFractional = 2, PowGradientNaN = 4, PowGradientZero = 8, PowGradientOne = 16, PowGradientTwo = 32;

    /// <summary>The training kernels, in the order they join <see cref="All"/>.</summary>
    internal static IReadOnlyList<GcnKernelSource> Training => _training ??= BuildTraining();

    // Built on first use: the kernels read static fields declared further down (the attention backward's arguments).
    private static GcnKernelSource[]? _training;

    /// <summary>The flags of the power x^<paramref name="p"/> (pow_bwd_f32 reads them, as CUDA's PtxKernels.PowFlags).</summary>
    internal static uint PowGradientFlags(float p)
    {
        bool integer = float.IsFinite(p) && MathF.Floor(p) == p;
        uint flags = 0;
        if (integer && MathF.IEEERemainder(p, 2f) != 0f)
        {
            flags |= PowGradientOdd;
        }

        if (float.IsFinite(p) && !integer)
        {
            flags |= PowGradientFractional;
        }

        if (float.IsNaN(p))
        {
            flags |= PowGradientNaN;
        }

        flags |= p == 0f ? PowGradientZero : p == 1f ? PowGradientOne : p == 2f ? PowGradientTwo : 0u;
        return flags;
    }

    private static string TrainingText(FormattableString text) => text.ToString(CultureInfo.InvariantCulture);

    private static GcnKernelSource[] BuildTraining() =>
    [
        .. UnaryBackwardKernels(),
        ExtremumBackward(),
        PowBackward(),
        ClampBackward(),
        LeakyReluBackward(),
        WhereBackward(),
        GatedBackward(),
        GatedBackwardPacked(),
        SoftmaxBackward(),
        RmsNormBackward(),
        NormBackward(),
        SgdMomentum(),
        Adam(),
        Adam8(),
        MultiSumSquares(),
        MultiAdamW(),
        ClipFactor(),
        Dropout(),
        AddDropout(),
        SumRows(),
        SumColumnsStrided(),
        WholeSum(squares: false),
        SumBlocks(),
        WholeSum(squares: true),
        SumAxis(),
        SumAxisRows(),
        BroadcastAxis(),
        ScatterChain(),
        ScatterWalk(),
        NormStats(),
        NormStatsPart(),
        NormStatsFinal(),
        NormApply(),
        GroupReduce(),
        GroupReducePart(),
        GroupReduceFinal(),
        GroupScaleShift(),
        WeightNorm(),
        WeightNormBackward(),
        .. TrainingAttention(),
    ];

    // ---- The element-wise writer -------------------------------------------------------------------------------------

    /// <summary>
    /// An element-wise training kernel's shape: its buffers and values (in argument order, n after them), the buffers read
    /// for each element (in the order the body takes them), the buffers written; loads and stores gated by a bit of a
    /// value (an optional buffer, or one read only when accumulating); whether the body needs the element's index; the
    /// values the body divides by (their reciprocals made once); scalars read once from a buffer's first element; and
    /// whether the kernel takes the tail alone (one element a work-item).
    /// </summary>
    private sealed record TrainingShape(string[] Buffers, string[] Values, int[] Read, int[] Written)
    {
        public (int Buffer, int Value, int Bit)[] LoadIf { get; init; } = [];

        public (int Buffer, int Value, int Bit)[] StoreIf { get; init; } = [];

        public bool Index { get; init; }

        public bool Single { get; init; }

        public int[] Divisors { get; init; } = [];

        public int[] Scalars { get; init; } = [];
    }

    /// <summary>One element as a training body sees it: registers by buffer index, the values, pointers and reciprocals.</summary>
    private sealed class TrainingElement
    {
        /// <summary>The element of each buffer read (null for a buffer not read).</summary>
        public required string[] In { get; init; }

        /// <summary>The register the body sets for each buffer written (null for a buffer not written).</summary>
        public required string[] Out { get; init; }

        /// <summary>The values' scalar registers.</summary>
        public required string[] Values { get; init; }

        /// <summary>Each buffer's pointer pair.</summary>
        public required string[] Pointers { get; init; }

        /// <summary>The reciprocal (vector) of each value the shape divides by (null otherwise).</summary>
        public required string[] Reciprocals { get; init; }

        /// <summary>Each buffer's first element, read once into a vector register (null unless the shape asks).</summary>
        public required string[] Scalars { get; init; }

        /// <summary>The element's index (a vector register; empty unless the shape asks).</summary>
        public required string Index { get; init; }

        /// <summary>A label prefix unique to this body.</summary>
        public required string Label { get; init; }

        /// <summary>n.</summary>
        public required string Count { get; init; }
    }

    // A load or store gated by a value's bit: skipped (to `skip`) when the bit is clear.
    private static void TrainingGate(PointwiseWriter w, string value, int bit, string skip)
    {
        w.I(TrainingText($"s_bitcmp1_b32 {value}, {bit}"));
        w.I(TrainingText($"s_cbranch_scc0 {skip}"));
    }

    // The reciprocal LLVM's unsigned division by a uniform divisor takes (a scalar count of at least 1), made once into
    // the held lines, in a vector register.
    private static string TrainingReciprocal(PointwiseWriter w, string divisor)
    {
        string r = w.V(), negated = w.V(), t = w.V();
        w.Held.Append(TrainingText($"""
              v_cvt_f32_u32_e32 {r}, {divisor}
              v_rcp_iflag_f32_e32 {r}, {r}
              v_mul_f32_e32 {r}, 0x4f7ffffe, {r}
              v_cvt_u32_f32_e32 {r}, {r}
              v_mov_b32_e32 {negated}, {divisor}
              v_sub_u32_e32 {negated}, 0, {negated}
              v_mul_lo_u32 {t}, {negated}, {r}
              v_mul_hi_u32 {t}, {r}, {t}
              v_add_u32_e32 {r}, {r}, {t}

            """));
        return r;
    }

    // x / d and x mod d (x a vector, d a uniform scalar, r its reciprocal): a high product, then two corrections.
    private static (string Quotient, string Remainder) TrainingDivMod(PointwiseWriter w, string x, string d, string r)
    {
        string q = w.V(), m = w.V(), t = w.V(), next = w.V();
        w.I(TrainingText($"v_mul_hi_u32 {q}, {x}, {r}"));
        w.I(TrainingText($"v_mul_lo_u32 {t}, {q}, {d}"));
        w.I(TrainingText($"v_sub_u32_e32 {m}, {x}, {t}"));
        for (int pass = 0; pass < 2; pass++)
        {
            w.I(TrainingText($"v_add_u32_e32 {next}, 1, {q}"));
            w.I(TrainingText($"v_subrev_u32_e32 {t}, {d}, {m}"));
            w.I(TrainingText($"v_cmp_le_u32_e32 vcc, {d}, {m}"));
            w.I(TrainingText($"v_cndmask_b32_e32 {m}, {m}, {t}, vcc"));
            w.I(TrainingText($"v_cndmask_b32_e32 {q}, {q}, {next}, vcc"));
        }

        return (q, m);
    }

    // A lane mask of every lane when a value's bit is set, of none when it is clear (for a uniform choice per lane).
    private static string TrainingFlagMask(PointwiseWriter w, string value, int bit)
    {
        string mask = w.Mask();
        w.I(TrainingText($"s_bitcmp1_b32 {value}, {bit}"));
        w.I(TrainingText($"s_cselect_b64 {mask}, -1, 0"));
        return mask;
    }

    // One element-wise training kernel's text (see the file's remarks): GcnKernels.Pointwise.cs's frame, generalized.
    private static GcnKernelSource TrainingElementwise(string name, TrainingShape shape, Action<PointwiseWriter, TrainingElement> body)
    {
        var w = new PointwiseWriter();
        int buffers = shape.Buffers.Length;
        int dwords = 2 * buffers + shape.Values.Length + 1;
        string[] pointers = [.. Enumerable.Range(0, buffers).Select(b => TrainingText($"%s[{2 * b}:{2 * b + 1}]"))];
        string[] values = [.. Enumerable.Range(0, shape.Values.Length).Select(k => TrainingText($"%s{2 * buffers + k}"))];
        string count = TrainingText($"%s{dwords - 1}");
        var reciprocals = new string[values.Length];
        foreach (int d in shape.Divisors)
        {
            reciprocals[d] = TrainingReciprocal(w, values[d]);
        }

        var scalars = new string[buffers];
        foreach (int b in shape.Scalars)
        {
            string s = w.Mask(), v = w.V();
            w.Held.Append(TrainingText($"""
                  s_load_dword {RowWriter.Lo(s)}, {pointers[b]}, 0x0
                  v_mov_b32_e32 {v}, {RowWriter.Lo(s)}

                """));
            scalars[b] = v;
        }

        int bodies = 0, gates = 0;
        string Label() => TrainingText($".L{name}_g{gates++}");
        TrainingElement Element(string[] inputs, string[] outputs, string index) => new()
        {
            In = inputs, Out = outputs, Values = values, Pointers = pointers, Reciprocals = reciprocals, Scalars = scalars, Index = index,
            Label = TrainingText($".L{name}_b{bodies++}"), Count = count,
        };

        (int Value, int Bit)? GateOf((int Buffer, int Value, int Bit)[] gates, int buffer)
        {
            foreach (var g in gates)
            {
                if (g.Buffer == buffer)
                {
                    return (g.Value, g.Bit);
                }
            }

            return null;
        }

        // The main loop: a chunk of 256 elements a wave a pass, 4 a lane.
        var main = new StringBuilder();
        if (!shape.Single)
        {
            w.Into(main);
            w.I($"v_add_u32_e32 {Offset}, {ChunkBytes}, {LaneBytes}");
            var quads = new int[buffers];
            Array.Fill(quads, -1);
            foreach (int b in shape.Read)
            {
                quads[b] = w.Range(4);
                if (GateOf(shape.LoadIf, b) is { } gate)
                {
                    string skip = Label();
                    for (int e = 0; e < 4; e++)
                    {
                        w.I(TrainingText($"v_mov_b32_e32 %v{quads[b] + e}, 0"));
                    }

                    TrainingGate(w, values[gate.Value], gate.Bit, skip);
                    w.I(TrainingText($"global_load_dwordx4 %v[{quads[b]}:{quads[b] + 3}], {Offset}, {pointers[b]}"));
                    w.I(TrainingText($"{skip}:"));
                }
                else
                {
                    w.I(TrainingText($"global_load_dwordx4 %v[{quads[b]}:{quads[b] + 3}], {Offset}, {pointers[b]}"));
                }
            }

            string indexBase = "";
            if (shape.Index)
            {
                indexBase = w.V();
                w.I(TrainingText($"v_lshrrev_b32_e32 {indexBase}, 2, {Offset}"));
            }

            var results = new int[buffers];
            Array.Fill(results, -1);
            foreach (int b in shape.Written)
            {
                results[b] = w.Range(4);
            }

            for (int e = 0; e < 4; e++)
            {
                var inputs = new string[buffers];
                var outputs = new string[buffers];
                for (int b = 0; b < buffers; b++)
                {
                    inputs[b] = quads[b] >= 0 ? TrainingText($"%v{quads[b] + e}") : null!;
                    outputs[b] = results[b] >= 0 ? TrainingText($"%v{results[b] + e}") : null!;
                }

                string index = indexBase;
                if (shape.Index && e > 0)
                {
                    index = w.V();
                    w.I(TrainingText($"v_add_u32_e32 {index}, {e}, {indexBase}"));
                }

                body(w, Element(inputs, outputs, index));
            }

            foreach (int b in shape.Written)
            {
                string store = TrainingText($"global_store_dwordx4 {Offset}, %v[{results[b]}:{results[b] + 3}], {pointers[b]}");
                if (GateOf(shape.StoreIf, b) is { } gate)
                {
                    string skip = Label();
                    TrainingGate(w, values[gate.Value], gate.Bit, skip);
                    w.I(store);
                    w.I(TrainingText($"{skip}:"));
                }
                else
                {
                    w.I(store);
                }
            }
        }

        // The tail: one element a work-item, grid-stride, with the lanes past n off.
        var tail = new StringBuilder();
        w.Into(tail);
        w.I($"v_lshlrev_b32_e32 {TailOffset}, 2, {TailElement}");
        {
            var inputs = new string[buffers];
            var outputs = new string[buffers];
            foreach (int b in shape.Read)
            {
                inputs[b] = w.V();
                if (GateOf(shape.LoadIf, b) is { } gate)
                {
                    string skip = Label();
                    w.I(TrainingText($"v_mov_b32_e32 {inputs[b]}, 0"));
                    TrainingGate(w, values[gate.Value], gate.Bit, skip);
                    w.I(TrainingText($"global_load_dword {inputs[b]}, {TailOffset}, {pointers[b]}"));
                    w.I(TrainingText($"{skip}:"));
                }
                else
                {
                    w.I(TrainingText($"global_load_dword {inputs[b]}, {TailOffset}, {pointers[b]}"));
                }
            }

            foreach (int b in shape.Written)
            {
                outputs[b] = w.V();
            }

            body(w, Element(inputs, outputs, TailElement));
            foreach (int b in shape.Written)
            {
                string store = TrainingText($"global_store_dword {TailOffset}, {outputs[b]}, {pointers[b]}");
                if (GateOf(shape.StoreIf, b) is { } gate)
                {
                    string skip = Label();
                    TrainingGate(w, values[gate.Value], gate.Bit, skip);
                    w.I(store);
                    w.I(TrainingText($"{skip}:"));
                }
                else
                {
                    w.I(store);
                }
            }
        }

        // The head: directives, the arguments, the frame.
        var text = new StringBuilder();
        text.Append(CultureInfo.InvariantCulture, $".kernel {name}\n");
        foreach (string buffer in shape.Buffers)
        {
            text.Append(CultureInfo.InvariantCulture, $".arg {buffer}, 8, global\n");
        }

        foreach (string value in shape.Values)
        {
            text.Append(CultureInfo.InvariantCulture, $".arg {value}, 4, value\n");
        }

        text.Append(CultureInfo.InvariantCulture, $"""
            .arg n, 4, value
            .reqd_workgroup_size {PointwiseBlock}, 1, 1
            .user_sgpr dispatch_ptr
            .user_sgpr kernarg_segment_ptr
            .system_sgpr workgroup_id_x

            """);
        var head = new StringBuilder();
        w.Into(head);
        w.I($"s_load_dword {GridItems}, s[0:1], 0xc              ; the grid's work-items (the dispatch packet's grid_size_x)");
        for (int at = 0; at < dwords;)
        {
            int size = dwords - at >= 8 ? 8 : dwords - at >= 4 ? 4 : dwords - at >= 2 ? 2 : 1;
            w.I(size == 1
                ? TrainingText($"s_load_dword %s{at}, s[2:3], 0x{4 * at:x}")
                : TrainingText($"s_load_dwordx{size} %s[{at}:{at + size - 1}], s[2:3], 0x{4 * at:x}"));
            at += size;
        }

        w.I($"v_readfirstlane_b32 {WaveChunk}, v0                ; the wave's first work-item in its block");
        w.I($"s_lshl_b32 {Scratch}, s4, 8                        ; the block's first work-item");
        w.I($"v_add_u32_e32 {Item}, {Scratch}, v0");
        w.I($"v_and_b32_e32 {LaneBytes}, 63, v0");
        w.I($"v_lshlrev_b32_e32 {LaneBytes}, 4, {LaneBytes}         ; the lane's bytes in a chunk");
        w.I($"s_lshr_b32 {WaveChunk}, {WaveChunk}, 6");
        w.I($"s_lshl_b32 {Scratch}, s4, 2");
        w.I($"s_add_u32 {WaveChunk}, {Scratch}, {WaveChunk}       ; the wave's first chunk: its wave in the grid");
        w.I($"s_lshr_b32 {Waves}, {GridItems}, 6                  ; the grid's waves: chunks a pass");
        int[] moved = [.. shape.Read.Concat(shape.Written).Distinct()];
        if (shape.Single || moved.Length == 0)
        {
            w.I($"s_mov_b32 {Chunks}, 0                               ; one element a work-item: no chunks");
        }
        else
        {
            w.I($"s_and_b32 {Chunks}, {count}, -4                     ; n rounded down to 4");
            w.I(TrainingText($"s_mov_b32 {Misaligned}, %s{2 * moved[0]}"));
            foreach (int buffer in moved.Skip(1))
            {
                w.I(TrainingText($"s_or_b32 {Misaligned}, {Misaligned}, %s{2 * buffer}"));
            }

            w.I($"s_and_b32 {Misaligned}, {Misaligned}, 15            ; SCC: a buffer moved 4 at a time is not 16-byte aligned");
            w.I($"s_cselect_b32 {Chunks}, 0, {Chunks}                 ; then no chunk is whole: the tail takes every element");
            w.I($"s_lshr_b32 {Chunks}, {Chunks}, 8                    ; whole chunks of 256");
        }

        w.I($"s_lshl_b32 {ChunkBytes}, {WaveChunk}, 10            ; the wave's chunk's byte offset");
        w.I($"s_lshl_b32 {PassBytes}, {Waves}, 10                 ; bytes a pass");
        w.I($"s_lshl_b32 {TailFirst}, {Chunks}, 8                 ; the tail's first element");

        string label = $".L{name}";
        text.Append(head).Append(w.Held);
        if (!shape.Single)
        {
            text.Append(CultureInfo.InvariantCulture, $"""
                  s_cmp_lt_u32 {WaveChunk}, {Chunks}
                  s_cbranch_scc0 {label}_tail
                {label}_main:

                """);
            text.Append(main);
            text.Append(CultureInfo.InvariantCulture, $"""
                  s_add_u32 {WaveChunk}, {WaveChunk}, {Waves}
                  s_add_u32 {ChunkBytes}, {ChunkBytes}, {PassBytes}
                  s_cmp_lt_u32 {WaveChunk}, {Chunks}
                  s_cbranch_scc1 {label}_main

                """);
        }

        text.Append(CultureInfo.InvariantCulture, $"""
            {label}_tail:
              v_add_u32_e32 {TailElement}, {TailFirst}, {Item}
              v_cmp_gt_u32_e32 vcc, {count}, {TailElement}
              s_and_saveexec_b64 {SavedExec}, vcc
              s_cbranch_execz {label}_done
            {label}_tail_loop:

            """);
        text.Append(tail);
        text.Append(CultureInfo.InvariantCulture, $"""
              v_add_u32_e32 {TailElement}, {GridItems}, {TailElement}
              v_cmp_gt_u32_e32 vcc, {count}, {TailElement}
              s_and_b64 exec, exec, vcc
              s_cbranch_execnz {label}_tail_loop
            {label}_done:
              s_endpgm

            """);
        return new GcnKernelSource(name, Encoding.UTF8.GetBytes(text.ToString()));
    }

    // ---- Element-wise backward passes ----------------------------------------------------------------------------------

    // dx += dy · op'(x) (CpuBackend.UnaryBackward), x at 0, y at 1, dy at 2, dx at 3; each reads what its derivative needs.
    private static IEnumerable<GcnKernelSource> UnaryBackwardKernels()
    {
        TrainingShape FromY = new(["x", "y", "dy", "dx"], [], [1, 2, 3], [3]), FromX = new(["x", "y", "dy", "dx"], [], [0, 2, 3], [3]);
        yield return TrainingElementwise("sigmoid_bwd_f32", FromY, (w, e) =>
        {
            string t = w.V();
            w.I(TrainingText($"v_sub_f32_e32 {t}, 1.0, {e.In[1]}"));
            w.I(TrainingText($"v_mul_f32_e32 {t}, {e.In[1]}, {t}"));
            w.I(TrainingText($"v_fma_f32 {e.Out[3]}, {e.In[2]}, {t}, {e.In[3]}"));
        });
        yield return TrainingElementwise("tanh_bwd_f32", FromY, (w, e) =>
        {
            string t = w.V();
            w.I(TrainingText($"v_mul_f32_e32 {t}, {e.In[1]}, {e.In[1]}"));
            w.I(TrainingText($"v_sub_f32_e32 {t}, 1.0, {t}"));
            w.I(TrainingText($"v_fma_f32 {e.Out[3]}, {e.In[2]}, {t}, {e.In[3]}"));
        });
        yield return TrainingElementwise("relu_bwd_f32", FromX, (w, e) =>
        {
            string t = w.V();
            w.I(TrainingText($"v_cmp_lt_f32_e32 vcc, 0, {e.In[0]}                   ; x > 0: dy"));
            w.I(TrainingText($"v_cndmask_b32_e32 {t}, 0, {e.In[2]}, vcc"));
            w.I(TrainingText($"v_add_f32_e32 {e.Out[3]}, {e.In[3]}, {t}"));
        });
        yield return TrainingElementwise("square_bwd_f32", FromX, (w, e) =>
        {
            string t = w.V();
            w.I(TrainingText($"v_add_f32_e32 {t}, {e.In[0]}, {e.In[0]}"));
            w.I(TrainingText($"v_fma_f32 {e.Out[3]}, {t}, {e.In[2]}, {e.In[3]}"));
        });
        yield return TrainingElementwise("abs_bwd_f32", FromX, (w, e) =>
        {
            string p = w.V(), m = w.V();
            w.I(TrainingText($"v_cmp_lt_f32_e32 vcc, 0, {e.In[0]}"));
            w.I(TrainingText($"v_cndmask_b32_e32 {p}, 0, {e.In[2]}, vcc"));
            w.I(TrainingText($"v_cmp_gt_f32_e32 vcc, 0, {e.In[0]}"));
            w.I(TrainingText($"v_cndmask_b32_e32 {m}, 0, {e.In[2]}, vcc"));
            w.I(TrainingText($"v_sub_f32_e32 {p}, {p}, {m}"));
            w.I(TrainingText($"v_add_f32_e32 {e.Out[3]}, {e.In[3]}, {p}"));
        });
        yield return TrainingElementwise("exp_bwd_f32", FromY, (w, e) => w.I(TrainingText($"v_fma_f32 {e.Out[3]}, {e.In[2]}, {e.In[1]}, {e.In[3]}")));
        yield return TrainingElementwise("log_bwd_f32", FromX, (w, e) =>
        {
            string t = w.V();
            w.Div(t, e.In[2], e.In[0]);
            w.I(TrainingText($"v_add_f32_e32 {e.Out[3]}, {e.In[3]}, {t}"));
        });
        yield return TrainingElementwise("gelu_bwd_f32", FromX, (w, e) => GeluTanhGradient(w, e.In[0], e.In[2], e.In[3], e.Out[3]));
        yield return TrainingElementwise("gelu_erf_bwd_f32", FromX, (w, e) =>
        {
            string x = e.In[0], z = w.V(), erf = w.V(), a = w.V(), b = w.V(), m = w.V(), exp = w.V(), t = w.V();
            w.I(TrainingText($"v_mul_f32_e32 {z}, {PointwiseWriter.Literal(0.70710678f)}, {x}"));
            TrainingErf(w, erf, z);
            w.I(TrainingText($"v_add_f32_e32 {a}, 1.0, {erf}"));
            w.I(TrainingText($"v_mul_f32_e32 {a}, 0.5, {a}"));
            w.I(TrainingText($"v_mul_f32_e32 {b}, {PointwiseWriter.Literal(0.3989422804f)}, {x}"));
            w.I(TrainingText($"v_mul_f32_e32 {m}, -0.5, {x}"));
            w.I(TrainingText($"v_mul_f32_e32 {m}, {m}, {x}"));
            w.Exp(exp, m);
            w.I(TrainingText($"v_mul_f32_e32 {b}, {b}, {exp}"));
            w.I(TrainingText($"v_add_f32_e32 {a}, {a}, {b}"));
            w.I(TrainingText($"v_mul_f32_e32 {t}, {e.In[2]}, {a}"));
            w.I(TrainingText($"v_add_f32_e32 {e.Out[3]}, {e.In[3]}, {t}"));
        });
        yield return TrainingElementwise("sqrt_bwd_f32", FromY, (w, e) =>
        {
            string t = w.V(), q = w.V();
            w.I(TrainingText($"v_mul_f32_e32 {t}, 0.5, {e.In[2]}"));
            w.Div(q, t, e.In[1]);
            w.I(TrainingText($"v_add_f32_e32 {e.Out[3]}, {e.In[3]}, {q}"));
        });
        yield return TrainingElementwise("sin_bwd_f32", FromX, (w, e) =>
        {
            string c = w.V(), t = w.V();
            w.SinCos(c, e.In[0], cosine: true);
            w.I(TrainingText($"v_mul_f32_e32 {t}, {e.In[2]}, {c}"));
            w.I(TrainingText($"v_add_f32_e32 {e.Out[3]}, {e.In[3]}, {t}"));
        });
        yield return TrainingElementwise("cos_bwd_f32", FromX, (w, e) =>
        {
            string s = w.V(), t = w.V();
            w.SinCos(s, e.In[0], cosine: false);
            w.I(TrainingText($"v_mul_f32_e32 {t}, {e.In[2]}, {s}"));
            w.I(TrainingText($"v_sub_f32_e32 {e.Out[3]}, {e.In[3]}, {t}"));
        });
        yield return TrainingElementwise("silu_bwd_f32", FromX, (w, e) =>
        {
            string s = w.V(), t = w.V(), u = w.V();
            w.Sigmoid(s, e.In[0]);
            w.I(TrainingText($"v_sub_f32_e32 {t}, 1.0, {s}"));
            w.I(TrainingText($"v_mul_f32_e32 {t}, {e.In[0]}, {t}"));
            w.I(TrainingText($"v_add_f32_e32 {t}, 1.0, {t}"));
            w.I(TrainingText($"v_mul_f32_e32 {u}, {e.In[2]}, {s}"));
            w.I(TrainingText($"v_mul_f32_e32 {u}, {u}, {t}"));
            w.I(TrainingText($"v_add_f32_e32 {e.Out[3]}, {e.In[3]}, {u}"));
        });
    }

    // erf z as the CPU computes it (CpuMath.Erf: Abramowitz and Stegun 7.1.26), as GELU's erf form does.
    private static void TrainingErf(PointwiseWriter w, string target, string z)
    {
        string a = w.V(), d = w.V(), t = w.V(), poly = w.V(), m = w.V(), e = w.V(), erf = w.V(), negated = w.V();
        w.I(TrainingText($"v_and_b32_e32 {a}, 0x7fffffff, {z}"));
        w.I(TrainingText($"v_fma_f32 {d}, {a}, {w.S(0.3275911f)}, 1.0"));
        w.Div(t, "1.0", d);
        w.I(TrainingText($"v_fma_f32 {poly}, {t}, {w.S(1.061405429f)}, {w.VC(-1.453152027f)}"));
        w.I(TrainingText($"v_fma_f32 {poly}, {poly}, {t}, {w.S(1.421413741f)}"));
        w.I(TrainingText($"v_fma_f32 {poly}, {poly}, {t}, {w.S(-0.284496736f)}"));
        w.I(TrainingText($"v_fma_f32 {poly}, {poly}, {t}, {w.S(0.254829592f)}"));
        w.I(TrainingText($"v_mul_f32_e32 {poly}, {poly}, {t}"));
        w.I(TrainingText($"v_mul_f32_e64 {m}, -{a}, {a}"));
        w.Exp(e, m);
        w.I(TrainingText($"v_mul_f32_e32 {poly}, {poly}, {e}"));
        w.I(TrainingText($"v_sub_f32_e32 {erf}, 1.0, {poly}"));
        w.I(TrainingText($"v_xor_b32_e32 {negated}, 0x80000000, {erf}"));
        w.I(TrainingText($"v_cmp_gt_f32_e32 vcc, 0, {z}                      ; z < 0: -erf"));
        w.I(TrainingText($"v_cndmask_b32_e32 {target}, {erf}, {negated}, vcc"));
    }

    // dx = fma(dy, GELU'(x), dx) for GELU's tanh form, as the CPU's vector path (GeluBackwardLoop): t = tanh(k fma(c x², x,
    // x)), GELU' = 0.5 (1 + t) + 0.5 x (1 - t²) k fma(3c, x², 1).
    private static void GeluTanhGradient(PointwiseWriter w, string x, string dy, string dx, string target)
    {
        const float K = 0.7978845608f, C = 0.044715f;
        string x2 = w.V(), c = w.V(), u = w.V(), t = w.V(), a = w.V(), b = w.V(), om = w.V(), f = w.V();
        w.I(TrainingText($"v_mul_f32_e32 {x2}, {x}, {x}"));
        w.I(TrainingText($"v_mul_f32_e32 {c}, {PointwiseWriter.Literal(C)}, {x2}"));
        w.I(TrainingText($"v_fma_f32 {u}, {c}, {x}, {x}"));
        w.I(TrainingText($"v_mul_f32_e32 {u}, {PointwiseWriter.Literal(K)}, {u}"));
        w.Tanh(t, u);
        w.I(TrainingText($"v_add_f32_e32 {a}, 1.0, {t}"));
        w.I(TrainingText($"v_mul_f32_e32 {a}, 0.5, {a}"));
        w.I(TrainingText($"v_mul_f32_e32 {b}, 0.5, {x}"));
        w.I(TrainingText($"v_mul_f32_e32 {om}, {t}, {t}"));
        w.I(TrainingText($"v_sub_f32_e32 {om}, 1.0, {om}"));
        w.I(TrainingText($"v_mul_f32_e32 {b}, {b}, {om}"));
        w.I(TrainingText($"v_mul_f32_e32 {b}, {PointwiseWriter.Literal(K)}, {b}"));
        w.I(TrainingText($"v_fma_f32 {f}, {x2}, {w.S(3f * C)}, 1.0"));
        w.I(TrainingText($"v_mul_f32_e32 {b}, {b}, {f}"));
        w.I(TrainingText($"v_add_f32_e32 {a}, {a}, {b}"));
        w.I(TrainingText($"v_fma_f32 {target}, {dy}, {a}, {dx}"));
    }

    // The gradient of max(a, b) or (flags bit 2) min(a, b): da += dy where a wins (a ≥ b, or a ≤ b; ties to a; never for
    // NaN), else db += dy; flags bit 0: da given, bit 1: db given (CpuBackend's ExtremumBackwardLoop, its vector form).
    private static GcnKernelSource ExtremumBackward() =>
        TrainingElementwise("extremum_bwd_f32", new(["a", "b", "dy", "da", "db"], ["flags"], [0, 1, 2, 3, 4], [3, 4])
        {
            LoadIf = [(3, 0, 0), (4, 0, 1)],
            StoreIf = [(3, 0, 0), (4, 0, 1)],
        }, (w, e) =>
        {
            string ge = w.Mask(), le = w.Mask(), first = w.Mask(), da = w.V(), db = w.V();
            w.I(TrainingText($"v_cmp_ge_f32_e64 {ge}, {e.In[0]}, {e.In[1]}"));
            w.I(TrainingText($"v_cmp_le_f32_e64 {le}, {e.In[0]}, {e.In[1]}"));
            w.I(TrainingText($"s_bitcmp1_b32 {e.Values[0]}, 2                     ; the minimum: a wins where a ≤ b"));
            w.I(TrainingText($"s_cselect_b64 {first}, {le}, {ge}"));
            w.I(TrainingText($"v_add_f32_e32 {da}, {e.In[3]}, {e.In[2]}"));
            w.I(TrainingText($"v_cndmask_b32_e64 {e.Out[3]}, {e.In[3]}, {da}, {first}"));
            w.I(TrainingText($"v_add_f32_e32 {db}, {e.In[4]}, {e.In[2]}"));
            w.I(TrainingText($"v_cndmask_b32_e64 {e.Out[4]}, {db}, {e.In[4]}, {first}"));
        });

    // da += dy where condition ≠ 0 (NaN too; flags bit 0: da given), db += dy elsewhere (bit 1: db given).
    private static GcnKernelSource WhereBackward() =>
        TrainingElementwise("where_bwd_f32", new(["condition", "dy", "da", "db"], ["flags"], [0, 1, 2, 3], [2, 3])
        {
            LoadIf = [(2, 0, 0), (3, 0, 1)],
            StoreIf = [(2, 0, 0), (3, 0, 1)],
        }, (w, e) =>
        {
            string second = w.Mask(), da = w.V(), db = w.V();
            w.I(TrainingText($"v_cmp_eq_f32_e64 {second}, 0, {e.In[0]}           ; ±0 selects b"));
            w.I(TrainingText($"v_add_f32_e32 {da}, {e.In[2]}, {e.In[1]}"));
            w.I(TrainingText($"v_cndmask_b32_e64 {e.Out[2]}, {da}, {e.In[2]}, {second}"));
            w.I(TrainingText($"v_add_f32_e32 {db}, {e.In[3]}, {e.In[1]}"));
            w.I(TrainingText($"v_cndmask_b32_e64 {e.Out[3]}, {e.In[3]}, {db}, {second}"));
        });

    // dx += dy where lo ≤ x ≤ hi (never for NaN).
    private static GcnKernelSource ClampBackward() =>
        TrainingElementwise("clamp_bwd_f32", new(["x", "dy", "dx"], ["lo", "hi"], [0, 1, 2], [2]), (w, e) =>
        {
            string above = w.Mask(), below = w.Mask(), sum = w.V();
            w.I(TrainingText($"v_cmp_le_f32_e64 {above}, {e.Values[0]}, {e.In[0]}"));
            w.I(TrainingText($"v_cmp_ge_f32_e64 {below}, {e.Values[1]}, {e.In[0]}"));
            w.I(TrainingText($"s_and_b64 {above}, {above}, {below}"));
            w.I(TrainingText($"v_add_f32_e32 {sum}, {e.In[2]}, {e.In[1]}"));
            w.I(TrainingText($"v_cndmask_b32_e64 {e.Out[2]}, {e.In[2]}, {sum}, {above}"));
        });

    // dx += dy where x > 0, slope · dy elsewhere.
    private static GcnKernelSource LeakyReluBackward() =>
        TrainingElementwise("leaky_relu_bwd_f32", new(["x", "dy", "dx"], ["slope"], [0, 1, 2], [2]), (w, e) =>
        {
            string t = w.V();
            w.I(TrainingText($"v_mul_f32_e32 {t}, {e.Values[0]}, {e.In[1]}"));
            w.I(TrainingText($"v_cmp_lt_f32_e32 vcc, 0, {e.In[0]}"));
            w.I(TrainingText($"v_cndmask_b32_e32 {t}, {t}, {e.In[1]}, vcc"));
            w.I(TrainingText($"v_add_f32_e32 {e.Out[2]}, {e.In[2]}, {t}"));
        });

    // dx += dy · e · x^p, p = e - 1 (flags: PowGradientFlags(p)), in the CPU's order. The host skips e = 0.
    private static GcnKernelSource PowBackward() =>
        TrainingElementwise("pow_bwd_f32", new(["x", "dy", "dx"], ["e", "p", "flags"], [0, 1, 2], [2]), (w, e) =>
        {
            string power = w.V(), t = w.V();
            TrainingPow(w, power, e.In[0], e.Values[1], e.Values[2]);
            w.I(TrainingText($"v_mul_f32_e32 {t}, {e.Values[0]}, {e.In[1]}"));
            w.I(TrainingText($"v_mul_f32_e32 {t}, {t}, {power}"));
            w.I(TrainingText($"v_add_f32_e32 {e.Out[2]}, {e.In[2]}, {t}"));
        });

    // target = x^p as MathF.Pow (CUDA's PowOf): |x|^p = 2^(p · log2|x|), the product's rounding error carried into the
    // exponential; denormal |x| scaled by 2^32 before the logarithm and results below 2^-126 scaled after the
    // exponential, as LLVM does; then MathF.Pow's special cases from the flags.
    private static void TrainingPow(PointwiseWriter w, string target, string x, string p, string flags)
    {
        string ax = w.V(), scaled = w.V(), lg = w.V(), shift = w.V(), t = w.V(), err = w.V(), small = w.V(), r = w.V(), corrected = w.V(), negated = w.V();
        string finite = w.Mask(), unit = w.Mask(), one = w.Mask(), flag = w.Mask(), sign = w.Mask(), negative = w.Mask();
        w.I(TrainingText($"v_and_b32_e32 {ax}, 0x7fffffff, {x}"));
        w.I(TrainingText($"v_cmp_gt_f32_e32 vcc, {w.S(0x00800000u)}, {ax}          ; below the smallest normal: scaled by 2^32"));
        w.I(TrainingText($"v_cndmask_b32_e32 {scaled}, 1.0, {w.VC(0x4f800000u)}, vcc"));
        w.I(TrainingText($"v_mul_f32_e32 {scaled}, {ax}, {scaled}"));
        w.I(TrainingText($"v_log_f32_e32 {lg}, {scaled}"));
        w.I(TrainingText($"v_cndmask_b32_e32 {shift}, 0, {w.VC(32f)}, vcc"));
        w.I(TrainingText($"v_sub_f32_e32 {lg}, {lg}, {shift}"));
        w.I(TrainingText($"v_mul_f32_e32 {t}, {p}, {lg}"));
        w.I(TrainingText($"v_fma_f32 {err}, {lg}, {p}, -{t}                     ; the product's rounding error"));
        w.I(TrainingText($"v_mul_f32_e32 {err}, {PointwiseWriter.Literal(0x3f317218u)}, {err}"));
        w.I(TrainingText($"v_cmp_gt_f32_e32 vcc, {w.S(-126f)}, {t}               ; below 2^-126: 2^(t + 64) · 2^-64"));
        w.I(TrainingText($"v_cndmask_b32_e32 {small}, 0, {w.VC(64f)}, vcc"));
        w.I(TrainingText($"v_add_f32_e32 {small}, {t}, {small}"));
        w.I(TrainingText($"v_exp_f32_e32 {r}, {small}"));
        w.I(TrainingText($"v_cndmask_b32_e32 {small}, 1.0, {w.VC(0x1f800000u)}, vcc"));
        w.I(TrainingText($"v_mul_f32_e32 {r}, {r}, {small}"));
        w.I(TrainingText($"v_cmp_lt_f32_e64 {finite}, |{t}|, {w.S(0x7f800000u)}"));
        w.I(TrainingText($"v_fma_f32 {corrected}, {r}, {err}, {r}"));
        w.I(TrainingText($"v_cndmask_b32_e64 {r}, {r}, {corrected}, {finite}"));
        w.I(TrainingText($"v_cmp_eq_f32_e64 {unit}, 1.0, {ax}                    ; |x| = 1: 1, but for a NaN power unless x = 1"));
        w.I(TrainingText($"v_cmp_eq_f32_e64 {one}, 1.0, {x}"));
        w.I(TrainingText($"s_bitcmp1_b32 {flags}, 2"));
        w.I(TrainingText($"s_cselect_b64 {unit}, {one}, {unit}"));
        w.I(TrainingText($"v_cndmask_b32_e64 {r}, {r}, 1.0, {unit}"));
        w.I(TrainingText($"v_cmp_gt_i32_e64 {sign}, 0, {x}                       ; a negative x to an odd power: negative"));
        w.I(TrainingText($"s_bitcmp1_b32 {flags}, 0"));
        w.I(TrainingText($"s_cselect_b64 {flag}, -1, 0"));
        w.I(TrainingText($"s_and_b64 {sign}, {sign}, {flag}"));
        w.I(TrainingText($"v_xor_b32_e32 {negated}, 0x80000000, {r}"));
        w.I(TrainingText($"v_cndmask_b32_e64 {r}, {r}, {negated}, {sign}"));
        w.I(TrainingText($"v_cmp_gt_f32_e64 {negative}, 0, {x}                  ; a finite negative x to a fractional power: NaN"));
        w.I(TrainingText($"v_cmp_lt_f32_e64 {finite}, |{x}|, {w.S(0x7f800000u)}"));
        w.I(TrainingText($"s_and_b64 {negative}, {negative}, {finite}"));
        w.I(TrainingText($"s_bitcmp1_b32 {flags}, 1"));
        w.I(TrainingText($"s_cselect_b64 {flag}, -1, 0"));
        w.I(TrainingText($"s_and_b64 {negative}, {negative}, {flag}"));
        w.I(TrainingText($"v_cndmask_b32_e64 {r}, {r}, {w.VC(0xffc00000u)}, {negative}"));
        w.I(TrainingText($"s_bitcmp1_b32 {flags}, 3                             ; p = 0: 1"));
        w.I(TrainingText($"s_cselect_b64 {flag}, -1, 0"));
        w.I(TrainingText($"v_cndmask_b32_e64 {r}, {r}, 1.0, {flag}"));
        w.I(TrainingText($"s_bitcmp1_b32 {flags}, 4                             ; p = 1: x"));
        w.I(TrainingText($"s_cselect_b64 {flag}, -1, 0"));
        w.I(TrainingText($"v_cndmask_b32_e64 {r}, {r}, {x}, {flag}"));
        w.I(TrainingText($"v_mul_f32_e32 {corrected}, {x}, {x}"));
        w.I(TrainingText($"s_bitcmp1_b32 {flags}, 5                             ; p = 2: x · x"));
        w.I(TrainingText($"s_cselect_b64 {flag}, -1, 0"));
        w.I(TrainingText($"v_cndmask_b32_e64 {target}, {r}, {corrected}, {flag}"));
    }

    // dgate += dy · up · act'(gate) (flags bit 0; bit 2: written instead), dup += dy · act(gate) (bit 1; bit 3: written);
    // kind 0 SiLU, 1 GELU (tanh form), 2 ReLU, 3 tanh(gate) · sigmoid(up) (CpuBackend.Quantized's Activation and UpSide).
    private static GcnKernelSource GatedBackward() =>
        TrainingElementwise("gated_act_bwd_f32", new(["gate", "up", "dy", "dgate", "dup"], ["kind", "flags"], [0, 1, 2, 3, 4], [3, 4])
        {
            LoadIf = [(3, 1, 0), (4, 1, 1)],
            StoreIf = [(3, 1, 0), (4, 1, 1)],
        }, (w, e) => GatedBackwardBody(w, e.Label, e.In[0], e.In[1], e.In[2], e.In[3], e.In[4], e.Values[0], e.Values[1], e.Out[3], e.Out[4]));

    // One element of the gated activation's gradient (gated_act_bwd_f32 and its packed form): from gate g, up u and dy, with
    // dgate's and dup's old values (0 where not given), into the written registers; `label` a prefix unique to the element.
    private static void GatedBackwardBody(PointwiseWriter w, string label, string g, string u, string dy, string oldGate, string oldUp, string kind, string flags,
        string newGate, string newUp)
    {
        const float K = 0.7978845608f, C = 0.044715f;
        string value = w.V(), slope = w.V(), upValue = w.V(), upSlope = w.V();
        string silu = label + "_silu", gelu = label + "_gelu", tanh = label + "_tanh", done = label + "_done";
        w.I(TrainingText($"""
            s_cmp_eq_u32 {kind}, 0
            s_cbranch_scc1 {silu}
            s_cmp_eq_u32 {kind}, 1
            s_cbranch_scc1 {gelu}
            s_cmp_eq_u32 {kind}, 3
            s_cbranch_scc1 {tanh}
            v_cmp_lt_f32_e32 vcc, 0, {g}
            v_cndmask_b32_e32 {value}, 0, {g}, vcc
            v_cndmask_b32_e64 {slope}, 0, 1.0, vcc
            v_mov_b32_e32 {upValue}, {u}
            v_mov_b32_e32 {upSlope}, 1.0
            s_branch {done}
            {silu}:
            """));
        string s = w.V(), t = w.V();
        w.Sigmoid(s, g);
        w.I(TrainingText($"""
            v_mul_f32_e32 {value}, {g}, {s}
            v_sub_f32_e32 {t}, 1.0, {s}
            v_mul_f32_e32 {t}, {g}, {t}
            v_add_f32_e32 {t}, 1.0, {t}
            v_mul_f32_e32 {slope}, {s}, {t}
            v_mov_b32_e32 {upValue}, {u}
            v_mov_b32_e32 {upSlope}, 1.0
            s_branch {done}
            {gelu}:
            """));
        string a = w.V(), th = w.V(), o = w.V(), h = w.V(), om = w.V(), b = w.V(), f = w.V();
        w.I(TrainingText($"""
            v_mul_f32_e32 {a}, {PointwiseWriter.Literal(C)}, {g}
            v_mul_f32_e32 {a}, {a}, {g}
            v_mul_f32_e32 {a}, {a}, {g}
            v_add_f32_e32 {a}, {g}, {a}
            v_mul_f32_e32 {a}, {PointwiseWriter.Literal(K)}, {a}
            """));
        w.Tanh(th, a);
        w.I(TrainingText($"""
            v_add_f32_e32 {o}, 1.0, {th}
            v_mul_f32_e32 {h}, 0.5, {g}
            v_mul_f32_e32 {value}, {h}, {o}
            v_mul_f32_e32 {o}, 0.5, {o}
            v_mul_f32_e32 {om}, {th}, {th}
            v_sub_f32_e32 {om}, 1.0, {om}
            v_mul_f32_e32 {b}, {h}, {om}
            v_mul_f32_e32 {b}, {PointwiseWriter.Literal(K)}, {b}
            v_mul_f32_e32 {f}, {PointwiseWriter.Literal(3f * C)}, {g}
            v_mul_f32_e32 {f}, {f}, {g}
            v_add_f32_e32 {f}, 1.0, {f}
            v_mul_f32_e32 {b}, {b}, {f}
            v_add_f32_e32 {slope}, {o}, {b}
            v_mov_b32_e32 {upValue}, {u}
            v_mov_b32_e32 {upSlope}, 1.0
            s_branch {done}
            {tanh}:
            """));
        string tg = w.V(), sg = w.V(), q = w.V();
        w.Tanh(tg, g);
        w.Sigmoid(sg, u);
        w.I(TrainingText($"""
            v_mov_b32_e32 {value}, {tg}
            v_mul_f32_e32 {q}, {tg}, {tg}
            v_sub_f32_e32 {slope}, 1.0, {q}
            v_mov_b32_e32 {upValue}, {sg}
            v_sub_f32_e32 {q}, 1.0, {sg}
            v_mul_f32_e32 {upSlope}, {sg}, {q}
            {done}:
            """));
        string written = TrainingFlagMask(w, flags, 2), dgate = w.V(), dup = w.V(), base1 = w.V(), base2 = w.V();
        w.I(TrainingText($"""
            v_cndmask_b32_e64 {base1}, {oldGate}, 0, {written}
            v_mul_f32_e32 {dgate}, {dy}, {upValue}
            v_mul_f32_e32 {dgate}, {dgate}, {slope}
            v_add_f32_e32 {newGate}, {base1}, {dgate}
            """));
        string written2 = TrainingFlagMask(w, flags, 3);
        w.I(TrainingText($"""
            v_cndmask_b32_e64 {base2}, {oldUp}, 0, {written2}
            v_mul_f32_e32 {dup}, {value}, {upSlope}
            v_mul_f32_e32 {dup}, {dy}, {dup}
            v_add_f32_e32 {newUp}, {base2}, {dup}
            """));
    }

    // gated_act_bwd_f32 with gate and up read as packed bfloat16 words (pg, pu: element 2p in word p's low half, 2p + 1 in
    // its high half), CUDA's gated_act_bwd_bf16_f32: a work-item a word (n = ⌈count / 2⌉ words), its two elements in turn
    // (the second while below count), dgate and dup read and written as gated_act_bwd_f32 reads and writes them.
    private static GcnKernelSource GatedBackwardPacked() =>
        TrainingElementwise("gated_act_bwd_bf16_f32", new(["pg", "pu", "dy", "dgate", "dup"], ["kind", "flags", "count"], [0, 1], [])
        {
            Index = true,
            Single = true,
        }, (w, e) =>
        {
            string kind = e.Values[0], flags = e.Values[1], count = e.Values[2];
            for (int half = 0; half < 2; half++)
            {
                string element = w.V(), at = w.V(), g = w.V(), u = w.V(), dy = w.V(), oldGate = w.V(), oldUp = w.V(), newGate = w.V(), newUp = w.V();
                string label = TrainingText($"{e.Label}_h{half}"), saved = w.Mask();
                w.I(TrainingText($"v_lshl_add_u32 {element}, {e.Index}, 1, {half}"));
                w.I(TrainingText($"v_cmp_gt_u32_e32 vcc, {count}, {element}"));
                w.I(TrainingText($"s_and_saveexec_b64 {saved}, vcc"));
                w.I(TrainingText($"s_cbranch_execz {label}_out"));
                w.I(TrainingText($"v_lshlrev_b32_e32 {at}, 2, {element}"));
                w.I(half == 0 ? TrainingText($"v_lshlrev_b32_e32 {g}, 16, {e.In[0]}") : TrainingText($"v_and_b32_e32 {g}, 0xffff0000, {e.In[0]}"));
                w.I(half == 0 ? TrainingText($"v_lshlrev_b32_e32 {u}, 16, {e.In[1]}") : TrainingText($"v_and_b32_e32 {u}, 0xffff0000, {e.In[1]}"));
                w.I(TrainingText($"global_load_dword {dy}, {at}, {e.Pointers[2]}"));
                w.I(TrainingText($"v_mov_b32_e32 {oldGate}, 0"));
                w.I(TrainingText($"v_mov_b32_e32 {oldUp}, 0"));
                TrainingGate(w, flags, 0, label + "_no_gate_in");
                w.I(TrainingText($"global_load_dword {oldGate}, {at}, {e.Pointers[3]}"));
                w.I(TrainingText($"{label}_no_gate_in:"));
                TrainingGate(w, flags, 1, label + "_no_up_in");
                w.I(TrainingText($"global_load_dword {oldUp}, {at}, {e.Pointers[4]}"));
                w.I(TrainingText($"{label}_no_up_in:"));
                GatedBackwardBody(w, label, g, u, dy, oldGate, oldUp, kind, flags, newGate, newUp);
                TrainingGate(w, flags, 0, label + "_no_gate_out");
                w.I(TrainingText($"global_store_dword {at}, {newGate}, {e.Pointers[3]}"));
                w.I(TrainingText($"{label}_no_gate_out:"));
                TrainingGate(w, flags, 1, label + "_no_up_out");
                w.I(TrainingText($"global_store_dword {at}, {newUp}, {e.Pointers[4]}"));
                w.I(TrainingText($"{label}_no_up_out:"));
                w.I(TrainingText($"{label}_out:"));
                w.I(TrainingText($"s_mov_b64 exec, {saved}"));
            }
        });

    // dx += invstd/m · (m · dxhat - s1 - xhat · s2) per group (g = i / inner mod groups) as the CPU's two affine passes:
    // dx += dxhat · invstd, then dx += fma(xhat, -invstd · s2 / m, -invstd · s1 / m).
    private static GcnKernelSource NormBackward() =>
        TrainingElementwise("norm_bwd_f32", new(["dxhat", "xhat", "s1", "s2", "invstd", "dx"], ["groups", "inner", "m"], [0, 1, 5], [5])
        {
            Index = true,
            Divisors = [0, 1],
        }, (w, e) =>
        {
            var (block, _) = TrainingDivMod(w, e.Index, e.Values[1], e.Reciprocals[1]);
            var (_, g) = TrainingDivMod(w, block, e.Values[0], e.Reciprocals[0]);
            string at = w.V(), s1 = w.V(), s2 = w.V(), inv = w.V(), negated = w.V(), m = w.V(), scale = w.V(), shift = w.V(), t = w.V(), u = w.V();
            w.I(TrainingText($"""
                v_lshlrev_b32_e32 {at}, 2, {g}
                global_load_dword {s1}, {at}, {e.Pointers[2]}
                global_load_dword {s2}, {at}, {e.Pointers[3]}
                global_load_dword {inv}, {at}, {e.Pointers[4]}
                v_mov_b32_e32 {m}, {e.Values[2]}
                v_xor_b32_e32 {negated}, 0x80000000, {inv}
                v_mul_f32_e32 {scale}, {negated}, {s2}
                v_mul_f32_e32 {shift}, {negated}, {s1}
                """));
            string scaleQ = w.V(), shiftQ = w.V();
            w.Div(scaleQ, scale, m);
            w.Div(shiftQ, shift, m);
            w.I(TrainingText($"""
                v_mul_f32_e32 {t}, {e.In[0]}, {inv}
                v_add_f32_e32 {t}, {e.In[5]}, {t}
                v_fma_f32 {u}, {e.In[1]}, {scaleQ}, {shiftQ}
                v_add_f32_e32 {e.Out[5]}, {t}, {u}
                """));
        });

    // ---- Optimizers and dropout ---------------------------------------------------------------------------------------

    // v = fma(momentum, v, g); p = fma(nlr, v, p) (nlr = -learning rate), the CPU's vector path.
    private static GcnKernelSource SgdMomentum() =>
        TrainingElementwise("sgd_momentum_f32", new(["p", "g", "v"], ["nlr", "momentum"], [0, 1, 2], [0, 2]), (w, e) =>
        {
            w.I(TrainingText($"v_fma_f32 {e.Out[2]}, {e.Values[1]}, {e.In[2]}, {e.In[1]}"));
            w.I(TrainingText($"v_fma_f32 {e.Out[0]}, {e.Values[0]}, {e.Out[2]}, {e.In[0]}"));
        });

    // m = fma(b1, m, c1 · g); v = fma(b2, v, c2 · g · g); p -= lr · m / (√v + eps) (the CPU's vector path, correctly rounded
    // square root and division).
    private static GcnKernelSource Adam() =>
        TrainingElementwise("adam_f32", new(["p", "g", "m", "v"], ["lr", "b1", "b2", "c1", "c2", "eps"], [0, 1, 2, 3], [0, 2, 3]), (w, e) =>
        {
            string t = w.V(), u = w.V(), root = w.V(), a = w.V(), q = w.V();
            w.I(TrainingText($"""
                v_mul_f32_e32 {t}, {e.Values[3]}, {e.In[1]}
                v_fma_f32 {e.Out[2]}, {e.Values[1]}, {e.In[2]}, {t}
                v_mul_f32_e32 {u}, {e.Values[4]}, {e.In[1]}
                v_mul_f32_e32 {u}, {u}, {e.In[1]}
                v_fma_f32 {e.Out[3]}, {e.Values[2]}, {e.In[3]}, {u}
                """));
            w.Sqrt(root, e.Out[3]);
            w.I(TrainingText($"v_add_f32_e32 {root}, {e.Values[5]}, {root}"));
            w.I(TrainingText($"v_mul_f32_e32 {a}, {e.Values[0]}, {e.Out[2]}"));
            w.Div(q, a, root);
            w.I(TrainingText($"v_sub_f32_e32 {e.Out[0]}, {e.In[0]}, {q}"));
        });

    // y = min(1, maxnorm / √x) for x > 0, else 1 (one value: the gradient clipping factor).
    private static GcnKernelSource ClipFactor() =>
        TrainingElementwise("clip_factor_f32", new(["x", "y"], ["maxnorm"], [0], [1]) { Single = true }, (w, e) =>
        {
            string root = w.V(), q = w.V(), least = w.V();
            w.Sqrt(root, e.In[0]);
            w.Div(q, w.VFrom(e.Values[0]), root);
            w.Extreme(least, w.VC(1f), q, minimum: true);
            w.I(TrainingText($"v_cmp_lt_f32_e32 vcc, 0, {e.In[0]}"));
            w.I(TrainingText($"v_cndmask_b32_e32 {e.Out[1]}, 1.0, {least}, vcc"));
        });

    // keep = DropoutMask.Keep(seed, i, p) (MurmurHash3's finalizer of i · 0x9E3779B9 ^ seed), bit for bit with the CPU.
    private static string DropoutKept(PointwiseWriter w, string index, string seed, string p)
    {
        string h = w.V(), t = w.V(), u = w.V();
        w.I(TrainingText($"""
            v_mul_lo_u32 {h}, {index}, {w.S(0x9E3779B9u)}
            v_xor_b32_e32 {h}, {seed}, {h}
            v_lshrrev_b32_e32 {t}, 16, {h}
            v_xor_b32_e32 {h}, {h}, {t}
            v_mul_lo_u32 {h}, {h}, {w.S(0x85EBCA6Bu)}
            v_lshrrev_b32_e32 {t}, 13, {h}
            v_xor_b32_e32 {h}, {h}, {t}
            v_mul_lo_u32 {h}, {h}, {w.S(0xC2B2AE35u)}
            v_lshrrev_b32_e32 {t}, 16, {h}
            v_xor_b32_e32 {h}, {h}, {t}
            v_lshrrev_b32_e32 {t}, 8, {h}
            v_cvt_f32_u32_e32 {u}, {t}
            v_mul_f32_e32 {u}, 0x33800000, {u}
            v_cmp_le_f32_e32 vcc, {p}, {u}
            """));
        return u;
    }

    // y = keep ? x · scale : 0, or y += it with accumulate (the backward pass); y is read only when accumulating.
    private static GcnKernelSource Dropout() =>
        TrainingElementwise("dropout_f32", new(["x", "y"], ["p", "scale", "seed", "accumulate"], [0, 1], [1])
        {
            Index = true,
            LoadIf = [(1, 3, 0)],
        }, (w, e) =>
        {
            string scaled = w.V(), kept = w.V(), sum = w.V();
            DropoutKept(w, e.Index, e.Values[2], e.Values[0]);
            w.I(TrainingText($"v_mul_f32_e32 {scaled}, {e.Values[1]}, {e.In[0]}"));
            w.I(TrainingText($"v_cndmask_b32_e32 {kept}, 0, {scaled}, vcc"));
            w.I(TrainingText($"v_add_f32_e32 {sum}, {e.In[1]}, {kept}"));
            string accumulate = TrainingFlagMask(w, e.Values[3], 0);
            w.I(TrainingText($"v_cndmask_b32_e64 {e.Out[1]}, {kept}, {sum}, {accumulate}"));
        });

    // y = r + (dropout_f32's mask with this seed)[i] · x · scale: a residual addition and dropout in one pass.
    private static GcnKernelSource AddDropout() =>
        TrainingElementwise("add_dropout_f32", new(["r", "x", "y"], ["p", "scale", "seed"], [0, 1], [2]) { Index = true }, (w, e) =>
        {
            string scaled = w.V(), kept = w.V();
            DropoutKept(w, e.Index, e.Values[2], e.Values[0]);
            w.I(TrainingText($"v_mul_f32_e32 {scaled}, {e.Values[1]}, {e.In[1]}"));
            w.I(TrainingText($"v_cndmask_b32_e32 {kept}, 0, {scaled}, vcc"));
            w.I(TrainingText($"v_add_f32_e32 {e.Out[2]}, {e.In[0]}, {kept}"));
        });

    // ---- Element-wise reductions' companions --------------------------------------------------------------------------

    // dx[o, d, j] += scale · dy[o, j] (i = (o · dim + d) · inner + j; diminner = dim · inner): the product, then the sum.
    private static GcnKernelSource BroadcastAxis() =>
        TrainingElementwise("broadcast_axis_f32", new(["dy", "dx"], ["diminner", "inner", "scale"], [1], [1])
        {
            Index = true,
            Divisors = [0, 1],
        }, (w, e) =>
        {
            var (outer, _) = TrainingDivMod(w, e.Index, e.Values[0], e.Reciprocals[0]);
            var (_, j) = TrainingDivMod(w, e.Index, e.Values[1], e.Reciprocals[1]);
            string at = w.V(), value = w.V(), t = w.V();
            w.I(TrainingText($"""
                v_mul_lo_u32 {at}, {outer}, {e.Values[1]}
                v_add_u32_e32 {at}, {at}, {j}
                v_lshlrev_b32_e32 {at}, 2, {at}
                global_load_dword {value}, {at}, {e.Pointers[0]}
                v_mul_f32_e32 {t}, {e.Values[2]}, {value}
                v_add_f32_e32 {e.Out[1]}, {e.In[1]}, {t}
                """));
        });

    // y = fma(x, invstd, -mean · invstd) per group (the CPU's NormApply: g = i / inner mod groups).
    private static GcnKernelSource NormApply() =>
        TrainingElementwise("norm_apply_f32", new(["x", "mean", "invstd", "y"], ["groups", "inner"], [0], [3])
        {
            Index = true,
            Divisors = [0, 1],
        }, (w, e) =>
        {
            var (block, _) = TrainingDivMod(w, e.Index, e.Values[1], e.Reciprocals[1]);
            var (_, g) = TrainingDivMod(w, block, e.Values[0], e.Reciprocals[0]);
            string at = w.V(), mean = w.V(), inv = w.V(), shift = w.V();
            w.I(TrainingText($"""
                v_lshlrev_b32_e32 {at}, 2, {g}
                global_load_dword {mean}, {at}, {e.Pointers[1]}
                global_load_dword {inv}, {at}, {e.Pointers[2]}
                v_xor_b32_e32 {shift}, 0x80000000, {mean}
                v_mul_f32_e32 {shift}, {shift}, {inv}
                v_fma_f32 {e.Out[3]}, {e.In[0]}, {inv}, {shift}
                """));
        });

    // y (+)= fma(x, scale[g], shift[g]); flags bit 0: a scale (else 1), bit 1: a shift (else 0), bit 2: accumulate.
    private static GcnKernelSource GroupScaleShift() =>
        TrainingElementwise("group_scale_shift_f32", new(["x", "scale", "shift", "y"], ["groups", "inner", "flags"], [0, 3], [3])
        {
            Index = true,
            Divisors = [0, 1],
            LoadIf = [(3, 2, 2)],
        }, (w, e) =>
        {
            var (block, _) = TrainingDivMod(w, e.Index, e.Values[1], e.Reciprocals[1]);
            var (_, g) = TrainingDivMod(w, block, e.Values[0], e.Reciprocals[0]);
            string at = w.V(), scale = w.V(), shift = w.V(), v = w.V(), sum = w.V();
            string noScale = e.Label + "_scaled", noShift = e.Label + "_shifted";
            w.I(TrainingText($"""
                v_lshlrev_b32_e32 {at}, 2, {g}
                v_mov_b32_e32 {scale}, 1.0
                v_mov_b32_e32 {shift}, 0
                s_bitcmp1_b32 {e.Values[2]}, 0
                s_cbranch_scc0 {noScale}
                global_load_dword {scale}, {at}, {e.Pointers[1]}
                {noScale}:
                s_bitcmp1_b32 {e.Values[2]}, 1
                s_cbranch_scc0 {noShift}
                global_load_dword {shift}, {at}, {e.Pointers[2]}
                {noShift}:
                v_fma_f32 {v}, {e.In[0]}, {scale}, {shift}
                v_add_f32_e32 {sum}, {e.In[3]}, {v}
                """));
            string accumulate = TrainingFlagMask(w, e.Values[2], 2);
            w.I(TrainingText($"v_cndmask_b32_e64 {e.Out[3]}, {v}, {sum}, {accumulate}"));
        });

    // y[o, j] (+)= scale · Σ_d x[o, d, j] (n = outer · inner): a work-item an output, the dimension in order (the CPU's
    // order where inner > 1), coalesced over j; the product, then the sum.
    private static GcnKernelSource SumAxis() =>
        TrainingElementwise("sum_axis_f32", new(["x", "y"], ["dim", "inner", "scale", "accumulate"], [1], [1])
        {
            Index = true,
            Single = true,
            Divisors = [1],
            LoadIf = [(1, 3, 0)],
        }, (w, e) =>
        {
            var (outer, j) = TrainingDivMod(w, e.Index, e.Values[1], e.Reciprocals[1]);
            string at = w.V(), step = w.V(), sum = w.V(), value = w.V(), d = w.Mask(), scaled = w.V(), added = w.V();
            string loop = e.Label + "_dims", end = e.Label + "_dims_end";
            w.I(TrainingText($"""
                v_mul_lo_u32 {at}, {outer}, {e.Values[0]}
                v_mul_lo_u32 {at}, {at}, {e.Values[1]}
                v_add_u32_e32 {at}, {at}, {j}
                v_lshlrev_b32_e32 {at}, 2, {at}
                v_mov_b32_e32 {step}, {e.Values[1]}
                v_lshlrev_b32_e32 {step}, 2, {step}
                v_mov_b32_e32 {sum}, 0
                s_mov_b32 {RowWriter.Lo(d)}, 0
                {loop}:
                s_cmp_ge_u32 {RowWriter.Lo(d)}, {e.Values[0]}
                s_cbranch_scc1 {end}
                global_load_dword {value}, {at}, {e.Pointers[0]}
                v_add_f32_e32 {sum}, {sum}, {value}
                v_add_u32_e32 {at}, {at}, {step}
                s_add_u32 {RowWriter.Lo(d)}, {RowWriter.Lo(d)}, 1
                s_branch {loop}
                {end}:
                v_mul_f32_e32 {scaled}, {e.Values[2]}, {sum}
                v_add_f32_e32 {added}, {e.In[1]}, {scaled}
                """));
            string accumulate = TrainingFlagMask(w, e.Values[3], 0);
            w.I(TrainingText($"v_cndmask_b32_e64 {e.Out[1]}, {scaled}, {added}, {accumulate}"));
        });

    // The second pass of scatter-add (n = count · dim): the work-item of each chain's head adds the id's rows of dy, in
    // position order, to the table's row (clamped to maxindex, as CUDA's), and stores it once.
    private static GcnKernelSource ScatterWalk() =>
        TrainingElementwise("scatter_walk_f32", new(["dy", "chain", "indices", "dtable"], ["dim", "maxindex", "count"], [], [])
        {
            Index = true,
            Single = true,
            Divisors = [0],
        }, (w, e) =>
        {
            var (position, column) = TrainingDivMod(w, e.Index, e.Values[0], e.Reciprocals[0]);
            string at = w.V(), link = w.V(), head = w.V(), id = w.V(), tableAt = w.V(), sum = w.V(), current = w.V(), rowAt = w.V(), value = w.V(), next = w.V();
            string saved = w.Mask(), heads = w.Mask();
            string walk = e.Label + "_walk", walked = e.Label + "_walked";
            w.I(TrainingText($"""
                v_lshlrev_b32_e32 {at}, 2, {position}
                global_load_dword {link}, {at}, {e.Pointers[1]}
                v_and_b32_e32 {head}, 0x80000000, {link}
                v_cmp_ne_u32_e32 vcc, 0, {head}
                s_and_saveexec_b64 {saved}, vcc
                s_mov_b64 {heads}, exec
                s_cbranch_execz {walked}
                global_load_dword {id}, {at}, {e.Pointers[2]}
                v_cvt_u32_f32_e32 {id}, {id}
                v_min_u32_e32 {id}, {e.Values[1]}, {id}
                v_mul_lo_u32 {tableAt}, {id}, {e.Values[0]}
                v_add_u32_e32 {tableAt}, {tableAt}, {column}
                v_lshlrev_b32_e32 {tableAt}, 2, {tableAt}
                global_load_dword {sum}, {tableAt}, {e.Pointers[3]}
                v_mov_b32_e32 {current}, {position}
                {walk}:
                v_mul_lo_u32 {rowAt}, {current}, {e.Values[0]}
                v_add_u32_e32 {rowAt}, {rowAt}, {column}
                v_lshlrev_b32_e32 {rowAt}, 2, {rowAt}
                global_load_dword {value}, {rowAt}, {e.Pointers[0]}
                v_lshlrev_b32_e32 {next}, 2, {current}
                global_load_dword {next}, {next}, {e.Pointers[1]}
                v_add_f32_e32 {sum}, {sum}, {value}
                v_and_b32_e32 {current}, 0x7fffffff, {next}
                v_cmp_gt_u32_e32 vcc, {e.Values[2]}, {current}
                s_and_b64 exec, exec, vcc
                s_cbranch_execnz {walk}
                s_mov_b64 exec, {heads}
                global_store_dword {tableAt}, {sum}, {e.Pointers[3]}
                {walked}:
                s_mov_b64 exec, {saved}
                """));
        });

    // ---- Helpers of the row kernels -------------------------------------------------------------------------------------

    // The element writer's math (correctly rounded division and square root, LLVM's exp) inside a row kernel: a fresh
    // PointwiseWriter, whose registers (%v16 up, %s64 up) are temporaries apart from the row writer's (%v100 up, %s100 up).
    // Each call's temporaries are renumbered into a range of their own (the writer's calls counted), so one virtual name
    // is never a single in one call and part of a pair in another (the allocator aligns a register's every range).
    private static string RowMathText(RowWriter w, Action<PointwiseWriter> math)
    {
        var pw = new PointwiseWriter();
        var body = new StringBuilder();
        pw.Into(body);
        math(pw);
        int shift;
        lock (RowMathCalls)
        {
            var calls = RowMathCalls.GetOrCreateValue(w);
            shift = 1000 + 200 * calls.Value++;
        }

        // Only the element writer's own temporaries (%v16 to %v99, %s64 to %s89): the row writer's registers (%v100 up,
        // %s90 up) passed in keep their names.
        string Renumber(System.Text.RegularExpressions.Match m)
        {
            bool vector = m.Groups[1].Value == "v";
            int first = int.Parse(m.Groups[m.Groups[2].Success ? 2 : 4].Value, CultureInfo.InvariantCulture);
            if (vector ? first is < 16 or >= 100 : first is < 64 or >= 90)
            {
                return m.Value;
            }

            return m.Groups[2].Success
                ? TrainingText($"%{m.Groups[1].Value}[{first + shift}:{int.Parse(m.Groups[3].Value, CultureInfo.InvariantCulture) + shift}]")
                : TrainingText($"%{m.Groups[1].Value}{first + shift}");
        }
        return RowMathRegister().Replace(pw.Held.ToString() + body, Renumber);
    }

    // %v16 / %s[64:65]: the element writer's virtual registers (its fixed frame, below 16 and 40, is never used by the math).
    [System.Text.RegularExpressions.GeneratedRegex(@"%([vs])(?:\[(\d+):(\d+)\]|(\d+))")]
    private static partial System.Text.RegularExpressions.Regex RowMathRegister();

    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<RowWriter, System.Runtime.CompilerServices.StrongBox<int>> RowMathCalls = [];

    private static void RowMath(RowWriter w, Action<PointwiseWriter> math) => w.L($"{RowMathText(w, math)}");

    // A range of `count` fresh scalar registers (for a wide scalar load); the first one's id.
    private static int RowRange(RowWriter w, int count)
    {
        string first = w.S();
        for (int i = 1; i < count; i++)
        {
            w.S();
        }

        return AttentionFirst(first);
    }

    // x / d and x mod d (x a vector, d a uniform scalar, r its reciprocal from AttentionReciprocal).
    private static (string Quotient, string Remainder) RowDivMod(RowWriter w, string x, string d, string r)
    {
        string q = w.V(), m = w.V(), t = w.V(), next = w.V();
        w.L($"""
            v_mul_hi_u32 {q}, {x}, {r}
            v_mul_lo_u32 {t}, {q}, {d}
            v_sub_u32_e32 {m}, {x}, {t}
            """);
        for (int pass = 0; pass < 2; pass++)
        {
            w.L($"""
                v_add_u32_e32 {next}, 1, {q}
                v_subrev_u32_e32 {t}, {d}, {m}
                v_cmp_le_u32_e32 vcc, {d}, {m}
                v_cndmask_b32_e32 {m}, {m}, {t}, vcc
                v_cndmask_b32_e32 {q}, {q}, {next}, vcc
                """);
        }

        return (q, m);
    }

    // The byte offset of element j of group g in the [outer, groups, inner] view: ((j / inner) · groups + g) · inner + j mod inner.
    private static string GroupIndex(RowWriter w, string j, string g, string groups, string inner, string byInner)
    {
        var (o, i) = RowDivMod(w, j, inner, byInner);
        string at = w.V();
        w.L($"""
            v_mul_lo_u32 {at}, {o}, {groups}
            v_add_u32_e32 {at}, {g}, {at}
            v_mul_lo_u32 {at}, {at}, {inner}
            v_add_u32_e32 {at}, {at}, {i}
            v_lshlrev_b32_e32 {at}, 2, {at}
            """);
        return at;
    }

    // A block's walk over `limit` elements from `start`: lane tid takes start + tid, start + tid + 256, ... in order (the
    // lanes past the limit off); `body` gets the element's index.
    private static void BlockElements(RowWriter w, string label, string start, string limit, Action<string> body)
    {
        string b = w.S(), j = w.V(), local = w.V(), saved = w.SPair();
        w.L($"s_mov_b32 {b}, 0");
        w.Label(label);
        w.L($"""
            s_cmp_ge_u32 {b}, {limit}
            s_cbranch_scc1 {label}_end
            v_add_u32_e32 {local}, {b}, {w.Tid}
            v_cmp_gt_u32_e32 vcc, {limit}, {local}
            s_and_saveexec_b64 {saved}, vcc
            v_add_u32_e32 {j}, {start}, {local}
            """);
        body(j);
        w.L($"""
            s_or_b64 exec, exec, {saved}
            s_add_u32 {b}, {b}, 0x100
            s_branch {label}
            """);
        w.Label(label + "_end");
    }

    // The larger of each pair (two reductions in one tree).
    private static string TrainingMax(RowWriter writer, string[] held, string[] other) => TrainingText($"""
        v_max_f32_e32 {held[0]}, {held[0]}, {other[0]}
        v_max_f32_e32 {held[1]}, {held[1]}, {other[1]}
        """);

    // (count, mean, M2) merged with another's (Chan et al., as CUDA's ChanMerge): n = na + nb, δ = mean_b − mean_a, mean =
    // mean_a + δ · nb / n, M2 = M2_a + (M2_b + δ² · na · nb / n); nothing when n = 0.
    private static string ChanMerge(RowWriter w, string na, string ma, string m2a, string nb, string mb, string m2b)
    {
        string n = w.V(), fa = w.V(), fb = w.V(), fn = w.V(), d = w.V(), r = w.V(), mean = w.V(), q = w.V();
        return TrainingText($"""
            v_add_u32_e32 {n}, {na}, {nb}
            v_cvt_f32_u32_e32 {fa}, {na}
            v_cvt_f32_u32_e32 {fb}, {nb}
            v_cvt_f32_u32_e32 {fn}, {n}
            v_sub_f32_e32 {d}, {mb}, {ma}

            """) + RowMathText(w, pw => pw.Div(r, fb, fn)) + TrainingText($"""
            v_fma_f32 {mean}, {d}, {r}, {ma}
            v_mul_f32_e32 {q}, {d}, {d}
            v_mul_f32_e32 {q}, {q}, {fa}
            v_fma_f32 {q}, {q}, {r}, {m2b}
            v_add_f32_e32 {q}, {m2a}, {q}
            v_cmp_ne_u32_e32 vcc, 0, {n}
            v_cndmask_b32_e32 {ma}, {ma}, {mean}, vcc
            v_cndmask_b32_e32 {m2a}, {m2a}, {q}, vcc
            v_mov_b32_e32 {na}, {n}
            """);
    }

    private static string ChanCombine(RowWriter writer, string[] held, string[] other) => ChanMerge(writer, held[0], held[1], held[2], other[0], other[1], other[2]);

    // ---- Row backward passes ------------------------------------------------------------------------------------------

    // Softmax: dx += y · (dy - Σ dy · y); log-softmax: dx += dy - e^y · Σ dy (the CPU's vector path), one block a row.
    private static GcnKernelSource SoftmaxBackward()
    {
        var w = new RowWriter("softmax_bwd_f32", arrays: 1);
        var a = w.Arguments(["y", "dy", "dx"], ["cols", "log", "n"]);
        string cols = a.Scalar("cols"), log = a.Scalar("log");
        w.BeginRows(a.Scalar("n"));
        int offset = w.RowOffset(cols);
        string yb = w.Base(a.Pointer("y"), offset), gb = w.Base(a.Pointer("dy"), offset), db = w.Base(a.Pointer("dx"), offset);
        var ys = w.Vs();
        var gs = w.Vs();
        var ds = w.Vs();
        var ts = w.Vs();
        string sum = w.V();
        w.L($"v_mov_b32_e32 {sum}, 0");
        w.L($"s_cmp_lg_u32 {log}, 0");
        w.L($"s_cbranch_scc1 .Lsum_log");
        w.Columns(cols, e => $"global_load_dword {ys[e.K]}, {e.Off}, {yb}{e.Imm}\nglobal_load_dword {gs[e.K]}, {e.Off}, {gb}{e.Imm}",
            e => $"v_fma_f32 {sum}, {gs[e.K]}, {ys[e.K]}, {sum}");
        w.L($"s_branch .Lsummed");
        w.Label(".Lsum_log");
        w.Columns(cols, e => $"global_load_dword {gs[e.K]}, {e.Off}, {gb}{e.Imm}", e => $"v_add_f32_e32 {sum}, {sum}, {gs[e.K]}");
        w.Label(".Lsummed");
        w.Reduce(RowWriter.Add, sum);
        string Loads(RowWriter.Element e) => $"""
            global_load_dword {ys[e.K]}, {e.Off}, {yb}{e.Imm}
            global_load_dword {gs[e.K]}, {e.Off}, {gb}{e.Imm}
            global_load_dword {ds[e.K]}, {e.Off}, {db}{e.Imm}
            """;
        w.L($"s_cmp_lg_u32 {log}, 0");
        w.L($"s_cbranch_scc1 .Lgrad_log");
        w.Columns(cols, Loads, e => $"""
            v_sub_f32_e32 {ts[e.K]}, {gs[e.K]}, {sum}
            v_mul_f32_e32 {ts[e.K]}, {ys[e.K]}, {ts[e.K]}
            v_add_f32_e32 {ds[e.K]}, {ds[e.K]}, {ts[e.K]}
            global_store_dword {e.Off}, {ds[e.K]}, {db}{e.Imm}
            """);
        w.L($"s_branch .Lgrad_done");
        w.Label(".Lgrad_log");
        w.Columns(cols, Loads, e => $"""
            v_mul_f32_e32 {ts[e.K]}, {RowWriter.Log2E}, {ys[e.K]}
            v_exp_f32_e32 {ts[e.K]}, {ts[e.K]}
            v_mul_f32_e32 {ts[e.K]}, {ts[e.K]}, {sum}
            v_sub_f32_e32 {ts[e.K]}, {gs[e.K]}, {ts[e.K]}
            v_add_f32_e32 {ds[e.K]}, {ds[e.K]}, {ts[e.K]}
            global_store_dword {e.Off}, {ds[e.K]}, {db}{e.Imm}
            """);
        w.Label(".Lgrad_done");
        w.EndRows();
        return w.Source();
    }

    // dx += inv · (dy - y · mean(dy · y)) per row (y the normalized output), one block a row.
    private static GcnKernelSource RmsNormBackward()
    {
        var w = new RowWriter("rms_norm_backward_f32", arrays: 1);
        var a = w.Arguments(["dy", "y", "inv", "dx"], ["cols", "n"]);
        string cols = a.Scalar("cols");
        string rc = w.ReciprocalOf(cols);
        w.BeginRows(a.Scalar("n"));
        int offset = w.RowOffset(cols);
        string gb = w.Base(a.Pointer("dy"), offset), yb = w.Base(a.Pointer("y"), offset), db = w.Base(a.Pointer("dx"), offset);
        string invAt = w.ElementAddress(a.Pointer("inv"), w.Row);
        var gs = w.Vs();
        var ys = w.Vs();
        var ds = w.Vs();
        var ts = w.Vs();
        string dot = w.V(), inv = w.V();
        w.L($"v_mov_b32_e32 {dot}, 0");
        w.L($"global_load_dword {inv}, {w.Zero}, {invAt}");
        w.Columns(cols, e => $"global_load_dword {gs[e.K]}, {e.Off}, {gb}{e.Imm}\nglobal_load_dword {ys[e.K]}, {e.Off}, {yb}{e.Imm}",
            e => $"v_fma_f32 {dot}, {gs[e.K]}, {ys[e.K]}, {dot}");
        w.Reduce(RowWriter.Add, dot);
        w.L($"v_mul_f32_e32 {dot}, {dot}, {rc}");
        w.Columns(cols,
            e => $"""
                global_load_dword {gs[e.K]}, {e.Off}, {gb}{e.Imm}
                global_load_dword {ys[e.K]}, {e.Off}, {yb}{e.Imm}
                global_load_dword {ds[e.K]}, {e.Off}, {db}{e.Imm}
                """,
            e => $"""
                v_mul_f32_e32 {ts[e.K]}, {ys[e.K]}, {dot}
                v_sub_f32_e32 {ts[e.K]}, {gs[e.K]}, {ts[e.K]}
                v_mul_f32_e32 {ts[e.K]}, {inv}, {ts[e.K]}
                v_add_f32_e32 {ds[e.K]}, {ds[e.K]}, {ts[e.K]}
                global_store_dword {e.Off}, {ds[e.K]}, {db}{e.Imm}
                """);
        w.EndRows();
        return w.Source();
    }

    // ---- Optimizers over blocks and chunks -------------------------------------------------------------------------

    // 8-bit Adam (Backend.AdamStep8Bit): one block a quantization block of 256 elements, one a thread. The two code maps
    // go to LDS once; the moments are decoded with the block's old scales, updated as the CPU updates them (correctly
    // rounded square root and division), the block's new scales are its maxima (one tree), and each moment is encoded to
    // its nearest code by binary search (EightBitMoments.Nearest). count = elements, n = blocks.
    private static GcnKernelSource Adam8()
    {
        var w = new RowWriter("adam8_f32", arrays: 2, extraLds: 2048);
        var a = w.Arguments(["p", "g", "m", "v", "absmax", "map"], ["lr", "b1", "b2", "c1", "c2", "eps", "count", "gscale", "decay", "n"]);
        string p = a.Pointer("p"), g = a.Pointer("g"), m = a.Pointer("m"), v = a.Pointer("v"), absmax = a.Pointer("absmax"), map = a.Pointer("map");
        string lr = a.Scalar("lr"), b1 = a.Scalar("b1"), b2 = a.Scalar("b2"), c1 = a.Scalar("c1"), c2 = a.Scalar("c2"), eps = a.Scalar("eps");
        string count = a.Scalar("count"), gscale = a.Scalar("gscale"), decay = a.Scalar("decay"), n = a.Scalar("n");
        int signedLds = w.ExtraLdsBase, unsignedLds = signedLds + 1024;
        string tid4 = w.V(), s0 = w.V(), u0 = w.V(), at = w.V();
        w.L($"""
            v_lshlrev_b32_e32 {tid4}, 2, {w.Tid}
            global_load_dword {s0}, {tid4}, {map}
            global_load_dword {u0}, {tid4}, {map} offset:1024
            v_add_u32_e32 {at}, 0x{signedLds:x}, {tid4}
            ds_write_b32 {at}, {s0}
            ds_write_b32 {at}, {u0} offset:1024
            s_barrier
            """);
        w.BeginRows(n);
        string first = w.S(), upper = w.S(), i = w.V(), off = w.V(), active = w.SPair(), saved = w.SPair();
        string mb = w.V(), vb = w.V(), gv = w.V(), pv = w.V(), mScale = w.V(), vScale = w.V();
        w.L($"""
            s_lshl_b32 {first}, {w.Row}, 8
            v_add_u32_e32 {i}, {first}, {w.Tid}
            v_lshlrev_b32_e32 {off}, 2, {i}
            s_add_u32 {upper}, {n}, {w.Row}
            """);
        string mAt = w.ElementAddress(absmax, w.Row), vAt = w.ElementAddress(absmax, upper);
        w.L($"""
            global_load_dword {mScale}, {w.Zero}, {mAt}
            global_load_dword {vScale}, {w.Zero}, {vAt}
            v_mov_b32_e32 {mb}, 0
            v_mov_b32_e32 {vb}, 0
            v_mov_b32_e32 {gv}, 0
            v_mov_b32_e32 {pv}, 0
            v_cmp_gt_u32_e64 {active}, {count}, {i}
            s_and_saveexec_b64 {saved}, {active}
            global_load_ubyte {mb}, {i}, {m}
            global_load_ubyte {vb}, {i}, {v}
            global_load_dword {gv}, {off}, {g}
            global_load_dword {pv}, {off}, {p}
            s_or_b64 exec, exec, {saved}
            """);
        string grad = w.V(), decayed = w.V(), mq = w.V(), vq = w.V(), t = w.V(), mom = w.V(), u = w.V(), vel = w.V();
        w.L($"""
            v_mul_f32_e32 {grad}, {gscale}, {gv}
            v_mul_f32_e32 {decayed}, {decay}, {pv}
            v_lshlrev_b32_e32 {t}, 2, {mb}
            ds_read_b32 {mq}, {t} offset:{signedLds}
            v_lshlrev_b32_e32 {u}, 2, {vb}
            ds_read_b32 {vq}, {u} offset:{unsignedLds}
            v_mul_f32_e32 {mq}, {mq}, {mScale}
            v_mul_f32_e32 {vq}, {vq}, {vScale}
            v_mul_f32_e32 {t}, {c1}, {grad}
            v_fma_f32 {mom}, {b1}, {mq}, {t}
            v_mul_f32_e32 {u}, {grad}, {grad}
            v_mul_f32_e32 {u}, {c2}, {u}
            v_fma_f32 {vel}, {b2}, {vq}, {u}
            """);
        string root = w.V(), q = w.V(), updated = w.V();
        RowMath(w, pw =>
        {
            pw.Sqrt(root, vel);
            pw.I(TrainingText($"v_add_f32_e32 {root}, {eps}, {root}"));
            pw.Div(q, mom, root);
        });
        w.L($"""
            v_mul_f32_e32 {q}, {lr}, {q}
            v_sub_f32_e32 {updated}, {decayed}, {q}
            s_and_saveexec_b64 {saved}, {active}
            global_store_dword {off}, {updated}, {p}
            s_or_b64 exec, exec, {saved}
            """);
        string mom0 = w.V(), vel0 = w.V(), am = w.V(), vm = w.V();
        w.L($"""
            v_cndmask_b32_e64 {mom0}, 0, {mom}, {active}
            v_cndmask_b32_e64 {vel0}, 0, {vel}, {active}
            v_and_b32_e32 {am}, 0x7fffffff, {mom0}
            v_mov_b32_e32 {vm}, {vel0}
            """);
        w.Reduce(TrainingMax, am, vm);
        string mn = w.V(), vn = w.V();
        RowMath(w, pw =>
        {
            pw.Div(mn, mom0, am);
            pw.Div(vn, vel0, vm);
        });
        w.L($"""
            v_cmp_lt_f32_e32 vcc, 0, {am}
            v_cndmask_b32_e32 {mn}, 0, {mn}, vcc
            v_cmp_lt_f32_e32 vcc, 0, {vm}
            v_cndmask_b32_e32 {vn}, 0, {vn}, vcc
            """);
        string mc = Adam8Encode(w, mn, signedLds), vc = Adam8Encode(w, vn, unsignedLds);
        w.L($"""
            s_and_saveexec_b64 {saved}, {active}
            global_store_byte {i}, {mc}, {m}
            global_store_byte {i}, {vc}, {v}
            s_or_b64 exec, exec, {saved}
            """);
        w.StoreFirstLane((mAt, am), (vAt, vm));
        w.EndRows();
        return w.Source();
    }

    // The index of the code nearest x in the sorted map of 256 codes at LDS `table` (EightBitMoments.Nearest).
    private static string Adam8Encode(RowWriter w, string x, int table)
    {
        string lo = w.V(), cand = w.V(), at = w.V(), value = w.V();
        w.L($"v_mov_b32_e32 {lo}, 0");
        for (int step = 128; step > 0; step >>= 1)
        {
            w.L($"""
                v_add_u32_e32 {cand}, 0x{step:x}, {lo}
                v_lshlrev_b32_e32 {at}, 2, {cand}
                ds_read_b32 {value}, {at} offset:{table}
                v_cmp_le_f32_e32 vcc, {value}, {x}
                v_cndmask_b32_e32 {lo}, {lo}, {cand}, vcc
                """);
        }

        string low = w.V(), high = w.V(), up = w.V(), d1 = w.V(), d2 = w.V(), next = w.V(), closer = w.SPair();
        w.L($"""
            v_lshlrev_b32_e32 {at}, 2, {lo}
            ds_read_b32 {low}, {at} offset:{table}
            v_min_u32_e32 {up}, 0xfe, {lo}
            v_lshlrev_b32_e32 {up}, 2, {up}
            ds_read_b32 {high}, {up} offset:{table + 4}
            v_sub_f32_e32 {d1}, {high}, {x}
            v_sub_f32_e32 {d2}, {x}, {low}
            v_cmp_lt_f32_e64 {closer}, {d1}, {d2}
            v_cmp_gt_u32_e32 vcc, 0xff, {lo}
            s_and_b64 vcc, vcc, {closer}
            v_add_u32_e32 {next}, 1, {lo}
            v_cndmask_b32_e32 {lo}, {lo}, {next}, vcc
            """);
        return lo;
    }

    // A chunk of the many-tensor optimizer (block = chunk): its table entry's address (40 bytes a tensor: the parameter,
    // gradient and moment pointers, then the count), its first element and its end (min(first + 4,096, count)).
    private static (string Entry, string First, string End) MultiChunk(RowWriter w, string chunks, string table)
    {
        string pair = w.SPair(), entry = w.SPair(), off = w.S(), count = w.S(), end = w.S();
        string at = w.Base(chunks, w.Product64(w.Row, "2"));
        w.L($"""
            s_load_dwordx2 {pair}, {at}, 0x0
            s_mul_i32 {off}, {RowWriter.Lo(pair)}, 40
            s_add_u32 {RowWriter.Lo(entry)}, {RowWriter.Lo(table)}, {off}
            s_addc_u32 {RowWriter.Hi(entry)}, {RowWriter.Hi(table)}, 0
            s_load_dword {count}, {entry}, 0x20
            s_add_u32 {end}, {RowWriter.Hi(pair)}, 0x{MultiAdamChunk:x}
            s_min_u32 {end}, {end}, {count}
            """);
        return (entry, RowWriter.Hi(pair), end);
    }

    // partials[chunk] = Σ g² over each chunk of every tensor's gradient (sum_blocks_f32 adds them in chunk order).
    private static GcnKernelSource MultiSumSquares()
    {
        var w = new RowWriter("multi_sumsq_f32", arrays: 1);
        var a = w.Arguments(["chunks", "table", "partials"], ["n"]);
        w.BeginRows(a.Scalar("n"));
        var (entry, first, end) = MultiChunk(w, a.Pointer("chunks"), a.Pointer("table"));
        string gp = w.SPair(), limit = w.S(), acc = w.V();
        w.L($"""
            s_load_dwordx2 {gp}, {entry}, 0x8
            s_sub_u32 {limit}, {end}, {first}
            v_mov_b32_e32 {acc}, 0
            """);
        BlockElements(w, ".Lsquares", first, limit, j =>
        {
            string off = w.V(), x = w.V();
            w.L($"""
                v_lshlrev_b32_e32 {off}, 2, {j}
                global_load_dword {x}, {off}, {gp}
                v_fma_f32 {acc}, {x}, {x}, {acc}
                """);
        });
        w.Reduce(RowWriter.Add, acc);
        w.StoreFirstLane((w.ElementAddress(a.Pointer("partials"), w.Row), acc));
        w.EndRows();
        return w.Source();
    }

    // AdamW over every tensor (the CPU's FusedAdamW): g · factor[0] (the clipping factor), the moments, p · decay - lr ·
    // m / (√v + eps); with zero ≠ 0 the gradient is zeroed for the next step.
    private static GcnKernelSource MultiAdamW()
    {
        var w = new RowWriter("multi_adamw_f32", arrays: 0);
        var a = w.Arguments(["chunks", "table", "factor"], ["lr", "decay", "b1", "b2", "c1", "c2", "eps", "zero", "n"]);
        string lr = a.Scalar("lr"), decay = a.Scalar("decay"), b1 = a.Scalar("b1"), b2 = a.Scalar("b2"), c1 = a.Scalar("c1"), c2 = a.Scalar("c2");
        string eps = a.Scalar("eps"), zero = a.Scalar("zero");
        string factor = w.V();
        w.L($"global_load_dword {factor}, {w.Zero}, {a.Pointer("factor")}");
        w.BeginRows(a.Scalar("n"));
        var (entry, first, end) = MultiChunk(w, a.Pointer("chunks"), a.Pointer("table"));
        int ptrs = RowRange(w, 8);
        string P(int k) => TrainingText($"%s[{ptrs + 2 * k}:{ptrs + 2 * k + 1}]");
        string limit = w.S();
        w.L($"""
            s_load_dwordx8 %s[{ptrs}:{ptrs + 7}], {entry}, 0x0
            s_sub_u32 {limit}, {end}, {first}
            """);
        BlockElements(w, ".Ladamw", first, limit, j =>
        {
            string off = w.V(), pv = w.V(), gv = w.V(), mv = w.V(), vv = w.V(), grad = w.V(), t = w.V(), mom = w.V(), u = w.V(), vel = w.V();
            string root = w.V(), step = w.V(), q = w.V(), pd = w.V(), pn = w.V(), nothing = w.V();
            w.L($"""
                v_lshlrev_b32_e32 {off}, 2, {j}
                global_load_dword {pv}, {off}, {P(0)}
                global_load_dword {gv}, {off}, {P(1)}
                global_load_dword {mv}, {off}, {P(2)}
                global_load_dword {vv}, {off}, {P(3)}
                v_mul_f32_e32 {grad}, {gv}, {factor}
                v_mul_f32_e32 {t}, {c1}, {grad}
                v_fma_f32 {mom}, {b1}, {mv}, {t}
                v_mul_f32_e32 {u}, {c2}, {grad}
                v_mul_f32_e32 {u}, {u}, {grad}
                v_fma_f32 {vel}, {b2}, {vv}, {u}
                v_mul_f32_e32 {step}, {lr}, {mom}
                """);
            RowMath(w, pw =>
            {
                pw.Sqrt(root, vel);
                pw.I(TrainingText($"v_add_f32_e32 {root}, {eps}, {root}"));
                pw.Div(q, step, root);
            });
            w.L($"""
                v_mul_f32_e32 {pd}, {decay}, {pv}
                v_sub_f32_e32 {pn}, {pd}, {q}
                global_store_dword {off}, {mom}, {P(2)}
                global_store_dword {off}, {vel}, {P(3)}
                global_store_dword {off}, {pn}, {P(0)}
                s_cmp_eq_u32 {zero}, 0
                s_cbranch_scc1 .Ladamw_kept
                v_mov_b32_e32 {nothing}, 0
                global_store_dword {off}, {nothing}, {P(1)}
                .Ladamw_kept:
                """);
        });
        w.EndRows();
        return w.Source();
    }

    // ---- Sums ----------------------------------------------------------------------------------------------------------

    // y[j] += Σ_r x[r, j] (few rows): a thread a column, the rows added in order onto y (the CPU's order: bit for bit).
    private static GcnKernelSource SumRows()
    {
        var w = new RowWriter("sum_rows_f32", arrays: 0);
        var a = w.Arguments(["x", "y"], ["rows", "cols"]);
        string x = a.Pointer("x"), y = a.Pointer("y"), rows = a.Scalar("rows"), cols = a.Scalar("cols");
        string grid = w.S(), stride = w.S(), block = w.S();
        w.L($"s_load_dword {grid}, s[0:1], 0xc                  ; grid_size_x");
        string rowBytes = w.Bytes64(cols);
        w.L($"s_lshr_b32 {stride}, {grid}, 8");
        w.L($"s_mov_b32 {block}, s4");
        w.Label(".Lblocks");
        string column0 = w.S(), column = w.V(), off = w.V(), saved = w.SPair(), acc = w.V(), value = w.V(), at = w.SPair(), r = w.S();
        w.L($"""
            s_lshl_b32 {column0}, {block}, 8
            s_cmp_ge_u32 {column0}, {cols}
            s_cbranch_scc1 .Ldone
            v_add_u32_e32 {column}, {column0}, {w.Tid}
            v_lshlrev_b32_e32 {off}, 2, {column}
            v_cmp_gt_u32_e32 vcc, {cols}, {column}
            s_and_saveexec_b64 {saved}, vcc
            global_load_dword {acc}, {off}, {y}
            s_mov_b64 {at}, {x}
            s_mov_b32 {r}, 0
            """);
        w.Label(".Lrows");
        w.L($"""
            s_cmp_ge_u32 {r}, {rows}
            s_cbranch_scc1 .Lrows_end
            global_load_dword {value}, {off}, {at}
            v_add_f32_e32 {acc}, {acc}, {value}
            """);
        w.Add64(at, rowBytes);
        w.L($"""
            s_add_u32 {r}, {r}, 1
            s_branch .Lrows
            """);
        w.Label(".Lrows_end");
        w.L($"""
            global_store_dword {off}, {acc}, {y}
            s_or_b64 exec, exec, {saved}
            s_add_u32 {block}, {block}, {stride}
            s_branch .Lblocks
            """);
        w.Label(".Ldone");
        w.L($"s_endpgm");
        return w.Source();
    }

    // y[j] += Σ_r x[r · ld + j]: a thread a column (grid x, grid-stride), grid y the chunks of `chunk` rows; one chunk adds
    // to y, several store partials[chunk · cols + j] for fold_chunks_f32 (rule 86: no atomics).
    private static GcnKernelSource SumColumnsStrided()
    {
        var w = new RowWriter("sum_cols_strided_f32", arrays: 0, workgroupY: true);
        var a = w.Arguments(["x", "y", "partials"], ["rows", "cols", "ld", "chunk"]);
        string x = a.Pointer("x"), y = a.Pointer("y"), partials = a.Pointer("partials");
        string rows = a.Scalar("rows"), cols = a.Scalar("cols"), ld = a.Scalar("ld"), chunk = a.Scalar("chunk");
        string grid = w.SPair(), stride = w.S(), first = w.S(), last = w.S(), block = w.S();
        string chunks = RowWriter.Hi(grid);
        w.L($"s_load_dwordx2 {grid}, s[0:1], 0xc                ; grid_size_x, grid_size_y (the chunks)");
        w.L($"s_lshr_b32 {stride}, {RowWriter.Lo(grid)}, 8");
        w.L($"s_mul_i32 {first}, s5, {chunk}");
        w.L($"s_add_u32 {last}, {first}, {chunk}");
        w.L($"s_min_u32 {last}, {last}, {rows}");
        string rowBytes = w.Bytes64(ld);
        string xb = w.Base(x, w.Product64(first, ld));
        string pb = w.Base(partials, w.Product64("s5", cols));
        w.L($"s_mov_b32 {block}, s4");
        w.Label(".Lblocks");
        string column0 = w.S(), column = w.V(), off = w.V(), saved = w.SPair(), acc = w.V(), value = w.V(), at = w.SPair(), r = w.S();
        w.L($"""
            s_lshl_b32 {column0}, {block}, 8
            s_cmp_ge_u32 {column0}, {cols}
            s_cbranch_scc1 .Ldone
            v_add_u32_e32 {column}, {column0}, {w.Tid}
            v_lshlrev_b32_e32 {off}, 2, {column}
            v_cmp_gt_u32_e32 vcc, {cols}, {column}
            s_and_saveexec_b64 {saved}, vcc
            v_mov_b32_e32 {acc}, 0
            s_mov_b64 {at}, {xb}
            s_mov_b32 {r}, {first}
            """);
        w.Label(".Lrows");
        w.L($"""
            s_cmp_ge_u32 {r}, {last}
            s_cbranch_scc1 .Lrows_end
            global_load_dword {value}, {off}, {at}
            v_add_f32_e32 {acc}, {acc}, {value}
            """);
        w.Add64(at, rowBytes);
        w.L($"""
            s_add_u32 {r}, {r}, 1
            s_branch .Lrows
            """);
        w.Label(".Lrows_end");
        w.L($"""
            s_cmp_eq_u32 {chunks}, 1
            s_cbranch_scc0 .Lpartial
            global_load_dword {value}, {off}, {y}
            v_add_f32_e32 {value}, {value}, {acc}
            global_store_dword {off}, {value}, {y}
            s_branch .Lnext
            """);
        w.Label(".Lpartial");
        w.L($"global_store_dword {off}, {acc}, {pb}");
        w.Label(".Lnext");
        w.L($"""
            s_or_b64 exec, exec, {saved}
            s_add_u32 {block}, {block}, {stride}
            s_branch .Lblocks
            """);
        w.Label(".Ldone");
        w.L($"s_endpgm");
        return w.Source();
    }

    // sum_f32: out[0] = scale · Σ x (one block), or partials[block] = the block's Σ (several; sum_blocks_f32 adds them in
    // block order); blocks = the grid, fixed by n. sumsq_f32: total[0] += Σ x² (direct, one block) or partials[block]; n =
    // the blocks. Each thread adds its elements block · 256 + tid + k · (blocks · 256) in order, four loads in flight, then
    // the block's tree.
    private static GcnKernelSource WholeSum(bool squares)
    {
        var w = new RowWriter(squares ? "sumsq_f32" : "sum_f32", arrays: 1);
        var a = squares ? w.Arguments(["x", "total", "partials"], ["count", "direct", "n"]) : w.Arguments(["x", "out", "partials"], ["n", "scale"]);
        string x = a.Pointer("x"), partials = a.Pointer("partials"), count = a.Scalar(squares ? "count" : "n");
        string blocks = w.S(), span = w.S(), span4 = w.S(), base0 = w.S(), end3 = w.S(), base4 = w.S(), tid4 = w.V(), acc = w.V();
        if (squares)
        {
            w.L($"s_mov_b32 {blocks}, {a.Scalar("n")}");
            w.L($"s_cmp_ge_u32 s4, {blocks}");
            w.L($"s_cbranch_scc1 .Ldone");
        }
        else
        {
            string grid = w.S();
            w.L($"s_load_dword {grid}, s[0:1], 0xc                ; grid_size_x");
            w.L($"s_lshr_b32 {blocks}, {grid}, 8");
        }

        w.L($"""
            s_lshl_b32 {span}, {blocks}, 8
            s_lshl_b32 {span4}, {span}, 2
            s_lshl_b32 {base0}, s4, 8
            v_lshlrev_b32_e32 {tid4}, 2, {w.Tid}
            v_mov_b32_e32 {acc}, 0
            """);
        string Add(string value) => squares ? TrainingText($"v_fma_f32 {acc}, {value}, {value}, {acc}") : TrainingText($"v_add_f32_e32 {acc}, {acc}, {value}");
        w.Label(".Lmain");
        var offs = new string[4];
        var vs = new string[4];
        for (int k = 0; k < 4; k++)
        {
            (offs[k], vs[k]) = (w.V(), w.V());
        }

        w.L($"""
            s_mul_i32 {end3}, {span}, 3
            s_add_u32 {end3}, {end3}, {base0}
            s_add_u32 {end3}, {end3}, 0x100
            s_cmp_gt_u32 {end3}, {count}
            s_cbranch_scc1 .Ltail
            s_lshl_b32 {base4}, {base0}, 2
            v_add_u32_e32 {offs[0]}, {base4}, {tid4}
            v_add_u32_e32 {offs[1]}, {span4}, {offs[0]}
            v_add_u32_e32 {offs[2]}, {span4}, {offs[1]}
            v_add_u32_e32 {offs[3]}, {span4}, {offs[2]}
            global_load_dword {vs[0]}, {offs[0]}, {x}
            global_load_dword {vs[1]}, {offs[1]}, {x}
            global_load_dword {vs[2]}, {offs[2]}, {x}
            global_load_dword {vs[3]}, {offs[3]}, {x}
            {Add(vs[0])}
            {Add(vs[1])}
            {Add(vs[2])}
            {Add(vs[3])}
            s_lshl_b32 {end3}, {span}, 2
            s_add_u32 {base0}, {base0}, {end3}
            s_branch .Lmain
            """);
        w.Label(".Ltail");
        string j = w.V(), off = w.V(), value = w.V(), saved = w.SPair();
        w.L($"""
            s_cmp_ge_u32 {base0}, {count}
            s_cbranch_scc1 .Lreduce
            v_add_u32_e32 {j}, {base0}, {w.Tid}
            v_cmp_gt_u32_e32 vcc, {count}, {j}
            s_and_saveexec_b64 {saved}, vcc
            v_lshlrev_b32_e32 {off}, 2, {j}
            global_load_dword {value}, {off}, {x}
            {Add(value)}
            s_or_b64 exec, exec, {saved}
            s_add_u32 {base0}, {base0}, {span}
            s_branch .Ltail
            """);
        w.Label(".Lreduce");
        string partAt = w.ElementAddress(partials, "s4");
        if (squares)
        {
            string total = a.Pointer("total"), before = w.V();
            w.L($"global_load_dword {before}, {w.Zero}, {total}");
            w.Reduce(RowWriter.Add, acc);
            w.L($"s_cmp_eq_u32 {a.Scalar("direct")}, 0");
            w.L($"s_cbranch_scc1 .Lpart");
            w.L($"v_add_f32_e32 {before}, {before}, {acc}");
            w.StoreFirstLane((total, before));
        }
        else
        {
            string scaled = w.V();
            w.Reduce(RowWriter.Add, acc);
            w.L($"s_cmp_lg_u32 {blocks}, 1");
            w.L($"s_cbranch_scc1 .Lpart");
            w.L($"v_mul_f32_e32 {scaled}, {a.Scalar("scale")}, {acc}");
            w.StoreFirstLane((a.Pointer("out"), scaled));
        }

        w.L($"s_branch .Ldone");
        w.Label(".Lpart");
        w.StoreFirstLane((partAt, acc));
        w.Label(".Ldone");
        w.L($"s_endpgm");
        return w.Source();
    }

    // out[0] = scale · Σ partials (accumulate: out[0] += it), one block: each thread adds partials tid, tid + 256, ... in
    // order, then the block's tree; the second pass of sum_f32, sumsq_f32 and multi_sumsq_f32.
    private static GcnKernelSource SumBlocks()
    {
        var w = new RowWriter("sum_blocks_f32", arrays: 1);
        var a = w.Arguments(["partials", "out"], ["count", "scale", "accumulate", "n"]);
        string partials = a.Pointer("partials"), output = a.Pointer("out");
        string acc = w.V(), before = w.V(), scaled = w.V(), sum = w.V(), mask = w.SPair();
        w.L($"""
            s_cmp_lg_u32 s4, 0
            s_cbranch_scc1 .Ldone
            v_mov_b32_e32 {acc}, 0
            global_load_dword {before}, {w.Zero}, {output}
            """);
        BlockElements(w, ".Lparts", "0", a.Scalar("count"), j =>
        {
            string off = w.V(), value = w.V();
            w.L($"""
                v_lshlrev_b32_e32 {off}, 2, {j}
                global_load_dword {value}, {off}, {partials}
                v_add_f32_e32 {acc}, {acc}, {value}
                """);
        });
        w.Reduce(RowWriter.Add, acc);
        w.L($"""
            v_mul_f32_e32 {scaled}, {a.Scalar("scale")}, {acc}
            v_add_f32_e32 {sum}, {before}, {scaled}
            s_cmp_lg_u32 {a.Scalar("accumulate")}, 0
            s_cselect_b64 {mask}, -1, 0
            v_cndmask_b32_e64 {scaled}, {scaled}, {sum}, {mask}
            """);
        w.StoreFirstLane((output, scaled));
        w.Label(".Ldone");
        w.L($"s_endpgm");
        return w.Source();
    }

    // y[o] (+)= scale · Σ_d x[o, d] (a sum along the last axis): one block a row, coalesced, the block's tree (n = outer).
    private static GcnKernelSource SumAxisRows()
    {
        var w = new RowWriter("sum_axis_rows_f32", arrays: 1);
        var a = w.Arguments(["x", "y"], ["dim", "scale", "accumulate", "n"]);
        string dim = a.Scalar("dim");
        w.BeginRows(a.Scalar("n"));
        string xb = w.Base(a.Pointer("x"), w.RowOffset(dim)), yAt = w.ElementAddress(a.Pointer("y"), w.Row);
        var xs = w.Vs();
        string acc = w.V(), before = w.V(), scaled = w.V(), sum = w.V(), mask = w.SPair();
        w.L($"v_mov_b32_e32 {acc}, 0");
        w.L($"global_load_dword {before}, {w.Zero}, {yAt}");
        w.Columns(dim, e => $"global_load_dword {xs[e.K]}, {e.Off}, {xb}{e.Imm}", e => $"v_add_f32_e32 {acc}, {acc}, {xs[e.K]}");
        w.Reduce(RowWriter.Add, acc);
        w.L($"""
            v_mul_f32_e32 {scaled}, {a.Scalar("scale")}, {acc}
            v_add_f32_e32 {sum}, {before}, {scaled}
            s_cmp_lg_u32 {a.Scalar("accumulate")}, 0
            s_cselect_b64 {mask}, -1, 0
            v_cndmask_b32_e64 {scaled}, {scaled}, {sum}, {mask}
            """);
        w.StoreFirstLane((yAt, scaled));
        w.EndRows();
        return w.Source();
    }

    // The first pass of scatter-add (see the file's remarks): chain[pos] = the next position with the same id (count when
    // none), its top bit set when no earlier position has the id (the chain's head). A block takes 256 positions (n = the
    // blocks, grid-stride); the ids come a tile of 256 at a time through LDS, every thread of the block at every barrier.
    private static GcnKernelSource ScatterChain()
    {
        var w = new RowWriter("scatter_chain_u32", arrays: 0, extraLds: 4 * RowThreads);
        var a = w.Arguments(["indices", "chain"], ["count", "maxindex", "n"]);
        string indices = a.Pointer("indices"), chain = a.Pointer("chain"), count = a.Scalar("count"), maxIndex = a.Scalar("maxindex");
        int idsLds = w.ExtraLdsBase;
        string tid4 = w.V(), writeAt = w.V();
        w.L($"""
            v_lshlrev_b32_e32 {tid4}, 2, {w.Tid}
            v_add_u32_e32 {writeAt}, 0x{idsLds:x}, {tid4}
            """);
        w.BeginRows(a.Scalar("n"));
        string first = w.S(), pos = w.V(), mine = w.SPair(), saved = w.SPair(), own = w.V(), off = w.V(), next = w.V(), head = w.V(), searching = w.SPair();
        string t0 = w.S();
        w.L($"""
            s_lshl_b32 {first}, {w.Row}, 8
            v_add_u32_e32 {pos}, {first}, {w.Tid}
            v_lshlrev_b32_e32 {off}, 2, {pos}
            v_mov_b32_e32 {own}, -1
            v_cmp_gt_u32_e64 {mine}, {count}, {pos}
            s_and_saveexec_b64 {saved}, {mine}
            global_load_dword {own}, {off}, {indices}
            v_cvt_u32_f32_e32 {own}, {own}
            v_min_u32_e32 {own}, {maxIndex}, {own}
            s_or_b64 exec, exec, {saved}
            v_mov_b32_e32 {next}, {count}
            v_mov_b32_e32 {head}, 0x80000000
            s_mov_b64 {searching}, {mine}
            s_mov_b32 {t0}, 0
            """);
        w.Label(".Ltile");
        string j = w.V(), id = w.V(), jOff = w.V(), tileCount = w.S(), full = w.SPair(), k = w.S(), kAt = w.V(), idk = w.V(), jk = w.S(), jv = w.V();
        string hit = w.SPair(), other = w.SPair(), earlier = w.SPair(), later = w.SPair();
        w.L($"""
            s_cmp_ge_u32 {t0}, {count}
            s_cbranch_scc1 .Llink
            v_add_u32_e32 {j}, {t0}, {w.Tid}
            v_lshlrev_b32_e32 {jOff}, 2, {j}
            v_mov_b32_e32 {id}, -1
            v_cmp_gt_u32_e32 vcc, {count}, {j}
            s_and_saveexec_b64 {saved}, vcc
            global_load_dword {id}, {jOff}, {indices}
            v_cvt_u32_f32_e32 {id}, {id}
            v_min_u32_e32 {id}, {maxIndex}, {id}
            s_or_b64 exec, exec, {saved}
            ds_write_b32 {writeAt}, {id}
            s_barrier
            s_sub_u32 {tileCount}, {count}, {t0}
            s_min_u32 {tileCount}, {tileCount}, 0x100
            s_mov_b64 {full}, exec
            s_mov_b64 exec, {searching}
            s_cbranch_execz .Lscanned
            s_mov_b32 {k}, 0
            v_mov_b32_e32 {kAt}, 0x{idsLds:x}
            """);
        w.Label(".Lscan");
        w.L($"""
            s_cmp_ge_u32 {k}, {tileCount}
            s_cbranch_scc1 .Lscanned
            ds_read_b32 {idk}, {kAt}
            s_add_u32 {jk}, {t0}, {k}
            v_cmp_eq_u32_e64 {hit}, {own}, {idk}
            v_cmp_ne_u32_e64 {other}, {jk}, {pos}
            s_and_b64 {hit}, {hit}, {other}
            v_cmp_lt_u32_e64 {earlier}, {jk}, {pos}
            s_and_b64 {other}, {hit}, {earlier}
            v_cndmask_b32_e64 {head}, {head}, 0, {other}
            s_andn2_b64 {later}, {hit}, {earlier}
            v_mov_b32_e32 {jv}, {jk}
            v_cndmask_b32_e64 {next}, {next}, {jv}, {later}
            v_add_u32_e32 {kAt}, 4, {kAt}
            s_add_u32 {k}, {k}, 1
            s_andn2_b64 exec, exec, {later}
            s_cbranch_execnz .Lscan
            """);
        w.Label(".Lscanned");
        w.L($"""
            s_mov_b64 {searching}, exec
            s_mov_b64 exec, {full}
            s_barrier
            s_add_u32 {t0}, {t0}, 0x100
            s_branch .Ltile
            """);
        w.Label(".Llink");
        string link = w.V();
        w.L($"""
            s_and_saveexec_b64 {saved}, {mine}
            v_or_b32_e32 {link}, {next}, {head}
            global_store_dword {off}, {link}, {chain}
            s_or_b64 exec, exec, {saved}
            """);
        w.EndRows();
        return w.Source();
    }

    // ---- Grouped normalization -----------------------------------------------------------------------------------------

    // Mean, variance and 1 / √(var + eps) of each group of the [outer, groups, inner] view: one block a group (grid-stride
    // over the groups), two passes (the mean, then the squares around it), each the block's tree.
    private static GcnKernelSource NormStats()
    {
        var w = new RowWriter("norm_stats_f32", arrays: 1);
        var a = w.Arguments(["x", "mean", "var", "invstd"], ["outer", "groups", "inner", "eps"]);
        string x = a.Pointer("x"), groups = a.Scalar("groups"), inner = a.Scalar("inner"), eps = a.Scalar("eps");
        string total = w.S(), totalF = w.V();
        w.L($"s_mul_i32 {total}, {a.Scalar("outer")}, {inner}");
        w.L($"v_cvt_f32_u32_e32 {totalF}, {total}");
        string byInner = AttentionReciprocal(w, inner);
        w.BeginRows(groups);
        string s1 = w.V(), s2 = w.V(), mean = w.V(), variance = w.V(), inv = w.V();
        w.L($"v_mov_b32_e32 {s1}, 0");
        BlockElements(w, ".Lfirst", "0", total, j =>
        {
            string at = GroupIndex(w, j, w.Row, groups, inner, byInner), value = w.V();
            w.L($"global_load_dword {value}, {at}, {x}");
            w.L($"v_add_f32_e32 {s1}, {s1}, {value}");
        });
        w.Reduce(RowWriter.Add, s1);
        RowMath(w, pw => pw.Div(mean, s1, totalF));
        w.L($"v_mov_b32_e32 {s2}, 0");
        BlockElements(w, ".Lsecond", "0", total, j =>
        {
            string at = GroupIndex(w, j, w.Row, groups, inner, byInner), value = w.V();
            w.L($"global_load_dword {value}, {at}, {x}");
            w.L($"v_sub_f32_e32 {value}, {value}, {mean}");
            w.L($"v_fma_f32 {s2}, {value}, {value}, {s2}");
        });
        w.Reduce(RowWriter.Add, s2);
        NormStatsStore(w, a, s2, totalF, eps, mean, variance, inv);
        w.EndRows();
        return w.Source();
    }

    // var = m2 / count, invstd = 1 / √(var + eps); the first thread stores the row's group's mean, var and invstd.
    private static void NormStatsStore(RowWriter w, RowWriter.Args a, string m2, string countF, string eps, string mean, string variance, string inv)
    {
        string shifted = w.V(), root = w.V();
        RowMath(w, pw =>
        {
            pw.Div(variance, m2, countF);
            pw.I(TrainingText($"v_add_f32_e32 {shifted}, {eps}, {variance}"));
            pw.Sqrt(root, shifted);
            pw.Div(inv, "1.0", root);
        });
        w.StoreFirstLane((w.ElementAddress(a.Pointer("mean"), w.Row), mean), (w.ElementAddress(a.Pointer("var"), w.Row), variance),
            (w.ElementAddress(a.Pointer("invstd"), w.Row), inv));
    }

    // The chunk of a split group kernel's block (row = group · chunks + chunk): its group, its first element and its count.
    private static (string Group, string First, string Count) GroupChunkOf(RowWriter w, string chunks, string byChunks, string total)
    {
        var (g, c) = AttentionDivMod(w, w.Row, chunks, byChunks);
        string first = w.S(), count = w.S();
        w.L($"""
            s_mul_i32 {first}, {c}, 0x{GroupChunk:x}
            s_sub_u32 {count}, {total}, {first}
            s_min_u32 {count}, {count}, 0x{GroupChunk:x}
            """);
        return (g, first, count);
    }

    // Each chunk's mean and the sum of squares around it (the chunk read twice); norm_stats_final_f32 merges them.
    // part: the chunks' means, then (after groups · chunks) their sums of squares.
    private static GcnKernelSource NormStatsPart()
    {
        var w = new RowWriter("norm_stats_part_f32", arrays: 1);
        var a = w.Arguments(["x", "part"], ["outer", "groups", "inner", "chunks"]);
        string x = a.Pointer("x"), groups = a.Scalar("groups"), inner = a.Scalar("inner"), chunks = a.Scalar("chunks");
        string total = w.S(), blocks = w.S();
        w.L($"s_mul_i32 {total}, {a.Scalar("outer")}, {inner}");
        w.L($"s_mul_i32 {blocks}, {groups}, {chunks}");
        string byInner = AttentionReciprocal(w, inner), byChunks = AttentionReciprocal(w, chunks);
        w.BeginRows(blocks);
        var (g, first, count) = GroupChunkOf(w, chunks, byChunks, total);
        string s1 = w.V(), s2 = w.V(), mean = w.V(), countF = w.V(), upper = w.S();
        w.L($"v_mov_b32_e32 {s1}, 0");
        w.L($"v_cvt_f32_u32_e32 {countF}, {count}");
        BlockElements(w, ".Lfirst", first, count, j =>
        {
            string at = GroupIndex(w, j, g, groups, inner, byInner), value = w.V();
            w.L($"global_load_dword {value}, {at}, {x}");
            w.L($"v_add_f32_e32 {s1}, {s1}, {value}");
        });
        w.Reduce(RowWriter.Add, s1);
        RowMath(w, pw => pw.Div(mean, s1, countF));
        w.L($"v_mov_b32_e32 {s2}, 0");
        BlockElements(w, ".Lsecond", first, count, j =>
        {
            string at = GroupIndex(w, j, g, groups, inner, byInner), value = w.V();
            w.L($"global_load_dword {value}, {at}, {x}");
            w.L($"v_sub_f32_e32 {value}, {value}, {mean}");
            w.L($"v_fma_f32 {s2}, {value}, {value}, {s2}");
        });
        w.Reduce(RowWriter.Add, s2);
        w.L($"s_add_u32 {upper}, {blocks}, {w.Row}");
        w.StoreFirstLane((w.ElementAddress(a.Pointer("part"), w.Row), mean), (w.ElementAddress(a.Pointer("part"), upper), s2));
        w.EndRows();
        return w.Source();
    }

    // The chunks' (count, mean, M2) merged per group: each thread its chunks tid, tid + 256, ... in order, then a tree of
    // the same merge; then the mean, the variance and 1 / √(var + eps).
    private static GcnKernelSource NormStatsFinal()
    {
        var w = new RowWriter("norm_stats_final_f32", arrays: 3);
        var a = w.Arguments(["part", "mean", "var", "invstd"], ["outer", "groups", "inner", "chunks", "eps"]);
        string part = a.Pointer("part"), groups = a.Scalar("groups"), chunks = a.Scalar("chunks");
        string total = w.S(), totalF = w.V(), blocks = w.S();
        w.L($"s_mul_i32 {total}, {a.Scalar("outer")}, {a.Scalar("inner")}");
        w.L($"s_mul_i32 {blocks}, {groups}, {chunks}");
        w.L($"v_cvt_f32_u32_e32 {totalF}, {total}");
        string second = w.Base(part, w.Product64(blocks, "1"));
        w.BeginRows(groups);
        string n = w.V(), mean = w.V(), m2 = w.V(), base0 = w.S(), variance = w.V(), inv = w.V();
        w.L($"""
            v_mov_b32_e32 {n}, 0
            v_mov_b32_e32 {mean}, 0
            v_mov_b32_e32 {m2}, 0
            s_mul_i32 {base0}, {w.Row}, {chunks}
            """);
        BlockElements(w, ".Lchunks", "0", chunks, c =>
        {
            string at = w.V(), mb = w.V(), m2b = w.V(), nb = w.V(), t = w.V();
            w.L($"""
                v_add_u32_e32 {at}, {base0}, {c}
                v_lshlrev_b32_e32 {at}, 2, {at}
                global_load_dword {mb}, {at}, {part}
                global_load_dword {m2b}, {at}, {second}
                v_lshlrev_b32_e32 {t}, 13, {c}
                v_sub_u32_e32 {nb}, {total}, {t}
                v_min_u32_e32 {nb}, 0x{GroupChunk:x}, {nb}
                """);
            w.L($"{ChanMerge(w, n, mean, m2, nb, mb, m2b)}");
        });
        w.Reduce(ChanCombine, n, mean, m2);
        NormStatsStore(w, a, m2, totalF, a.Scalar("eps"), mean, variance, inv);
        w.EndRows();
        return w.Source();
    }

    // sumA[g] += Σ a, sumAB[g] += Σ a · b (when hasb) per group: one block a group, one tree for both.
    private static GcnKernelSource GroupReduce()
    {
        var w = new RowWriter("group_reduce_f32", arrays: 2);
        var a = w.Arguments(["a", "b", "suma", "sumab"], ["outer", "groups", "inner", "hasb"]);
        string pa = a.Pointer("a"), pb = a.Pointer("b"), groups = a.Scalar("groups"), inner = a.Scalar("inner");
        string total = w.S();
        w.L($"s_mul_i32 {total}, {a.Scalar("outer")}, {inner}");
        string byInner = AttentionReciprocal(w, inner);
        w.BeginRows(groups);
        string s1 = w.V(), s2 = w.V();
        w.L($"v_mov_b32_e32 {s1}, 0");
        w.L($"v_mov_b32_e32 {s2}, 0");
        BlockElements(w, ".Lsum", "0", total, j =>
        {
            string at = GroupIndex(w, j, w.Row, groups, inner, byInner), av = w.V(), bv = w.V();
            w.L($"""
                global_load_dword {av}, {at}, {pa}
                global_load_dword {bv}, {at}, {pb}
                v_add_f32_e32 {s1}, {s1}, {av}
                v_fma_f32 {s2}, {av}, {bv}, {s2}
                """);
        });
        GroupSumsStore(w, a, w.Row, s1, s2);
        w.EndRows();
        return w.Source();
    }

    // The block's two sums added to suma[g] and (hasb) sumab[g] by the first thread (the old values read before the tree).
    private static void GroupSumsStore(RowWriter w, RowWriter.Args a, string g, string s1, string s2)
    {
        string sumAAt = w.ElementAddress(a.Pointer("suma"), g), sumABAt = w.ElementAddress(a.Pointer("sumab"), g), oldA = w.V(), oldAB = w.V();
        w.L($"global_load_dword {oldA}, {w.Zero}, {sumAAt}");
        w.L($"global_load_dword {oldAB}, {w.Zero}, {sumABAt}");
        w.Reduce(RowWriter.Add, s1, s2);
        w.L($"v_add_f32_e32 {oldA}, {oldA}, {s1}");
        w.L($"v_add_f32_e32 {oldAB}, {oldAB}, {s2}");
        w.StoreFirstLane((sumAAt, oldA));
        w.L($"s_cmp_eq_u32 {a.Scalar("hasb")}, 0");
        w.L($"s_cbranch_scc1 .Lno_b");
        w.StoreFirstLane((sumABAt, oldAB));
        w.Label(".Lno_b");
    }

    // Each chunk's Σ a and Σ a · b (when hasb) into part (the first sums, then after groups · chunks the second).
    private static GcnKernelSource GroupReducePart()
    {
        var w = new RowWriter("group_reduce_part_f32", arrays: 2);
        var a = w.Arguments(["a", "b", "part"], ["outer", "groups", "inner", "chunks", "hasb"]);
        string pa = a.Pointer("a"), pb = a.Pointer("b"), groups = a.Scalar("groups"), inner = a.Scalar("inner"), chunks = a.Scalar("chunks");
        string total = w.S(), blocks = w.S();
        w.L($"s_mul_i32 {total}, {a.Scalar("outer")}, {inner}");
        w.L($"s_mul_i32 {blocks}, {groups}, {chunks}");
        string byInner = AttentionReciprocal(w, inner), byChunks = AttentionReciprocal(w, chunks);
        w.BeginRows(blocks);
        var (g, first, count) = GroupChunkOf(w, chunks, byChunks, total);
        string s1 = w.V(), s2 = w.V(), upper = w.S();
        w.L($"v_mov_b32_e32 {s1}, 0");
        w.L($"v_mov_b32_e32 {s2}, 0");
        BlockElements(w, ".Lsum", first, count, j =>
        {
            string at = GroupIndex(w, j, g, groups, inner, byInner), av = w.V(), bv = w.V();
            w.L($"""
                global_load_dword {av}, {at}, {pa}
                global_load_dword {bv}, {at}, {pb}
                v_add_f32_e32 {s1}, {s1}, {av}
                v_fma_f32 {s2}, {av}, {bv}, {s2}
                """);
        });
        w.Reduce(RowWriter.Add, s1, s2);
        w.L($"s_add_u32 {upper}, {blocks}, {w.Row}");
        w.StoreFirstLane((w.ElementAddress(a.Pointer("part"), w.Row), s1), (w.ElementAddress(a.Pointer("part"), upper), s2));
        w.EndRows();
        return w.Source();
    }

    // Each group's chunks added in chunk order (each thread its chunks, then the tree) to suma and (hasb) sumab.
    private static GcnKernelSource GroupReduceFinal()
    {
        var w = new RowWriter("group_reduce_final_f32", arrays: 2);
        var a = w.Arguments(["part", "suma", "sumab"], ["groups", "chunks", "hasb"]);
        string part = a.Pointer("part"), groups = a.Scalar("groups"), chunks = a.Scalar("chunks");
        string blocks = w.S();
        w.L($"s_mul_i32 {blocks}, {groups}, {chunks}");
        string second = w.Base(part, w.Product64(blocks, "1"));
        w.BeginRows(groups);
        string s1 = w.V(), s2 = w.V(), base0 = w.S();
        w.L($"""
            v_mov_b32_e32 {s1}, 0
            v_mov_b32_e32 {s2}, 0
            s_mul_i32 {base0}, {w.Row}, {chunks}
            """);
        BlockElements(w, ".Lchunks", base0, chunks, j =>
        {
            string at = w.V(), p1 = w.V(), p2 = w.V();
            w.L($"""
                v_lshlrev_b32_e32 {at}, 2, {j}
                global_load_dword {p1}, {at}, {part}
                global_load_dword {p2}, {at}, {second}
                v_add_f32_e32 {s1}, {s1}, {p1}
                v_add_f32_e32 {s2}, {s2}, {p2}
                """);
        });
        GroupSumsStore(w, a, w.Row, s1, s2);
        w.EndRows();
        return w.Source();
    }

    // ---- Weight normalization -------------------------------------------------------------------------------------------

    // weight = gain[r] · direction / ‖direction‖ per row (one block a row; the norm correctly rounded, then the quotient).
    private static GcnKernelSource WeightNorm()
    {
        var w = new RowWriter("weight_norm_f32", arrays: 1);
        var a = w.Arguments(["gain", "direction", "weight"], ["cols", "n"]);
        string cols = a.Scalar("cols");
        w.BeginRows(a.Scalar("n"));
        int offset = w.RowOffset(cols);
        string vb = w.Base(a.Pointer("direction"), offset), wb = w.Base(a.Pointer("weight"), offset), gainAt = w.ElementAddress(a.Pointer("gain"), w.Row);
        var vs = w.Vs();
        var ws = w.Vs();
        string ss = w.V(), gain = w.V(), root = w.V(), scale = w.V();
        w.L($"v_mov_b32_e32 {ss}, 0");
        w.L($"global_load_dword {gain}, {w.Zero}, {gainAt}");
        w.Columns(cols, e => $"global_load_dword {vs[e.K]}, {e.Off}, {vb}{e.Imm}", e => $"v_fma_f32 {ss}, {vs[e.K]}, {vs[e.K]}, {ss}");
        w.Reduce(RowWriter.Add, ss);
        RowMath(w, pw =>
        {
            pw.Sqrt(root, ss);
            pw.Div(scale, gain, root);
        });
        w.Columns(cols, e => $"global_load_dword {vs[e.K]}, {e.Off}, {vb}{e.Imm}", e => $"""
            v_mul_f32_e32 {ws[e.K]}, {vs[e.K]}, {scale}
            global_store_dword {e.Off}, {ws[e.K]}, {wb}{e.Imm}
            """);
        w.EndRows();
        return w.Source();
    }

    // With s = Σ direction², dot = Σ dweight · direction per row: dgain[r] += dot / √s (flags bit 0), ddirection +=
    // gain / √s · dweight - gain / √s · dot / s · direction (bit 1), as the CPU's two fused multiply-adds.
    private static GcnKernelSource WeightNormBackward()
    {
        var w = new RowWriter("weight_norm_bwd_f32", arrays: 2);
        var a = w.Arguments(["gain", "direction", "dweight", "dgain", "ddirection"], ["cols", "flags", "n"]);
        string cols = a.Scalar("cols"), flags = a.Scalar("flags");
        w.BeginRows(a.Scalar("n"));
        int offset = w.RowOffset(cols);
        string vb = w.Base(a.Pointer("direction"), offset), gb = w.Base(a.Pointer("dweight"), offset), db = w.Base(a.Pointer("ddirection"), offset);
        string gainAt = w.ElementAddress(a.Pointer("gain"), w.Row), dgainAt = w.ElementAddress(a.Pointer("dgain"), w.Row);
        var vs = w.Vs();
        var gs = w.Vs();
        var ds = w.Vs();
        var ts = w.Vs();
        string ss = w.V(), dot = w.V(), gain = w.V(), dgain = w.V(), root = w.V(), inv = w.V(), scale = w.V(), along = w.V(), t = w.V();
        w.L($"""
            v_mov_b32_e32 {ss}, 0
            v_mov_b32_e32 {dot}, 0
            global_load_dword {gain}, {w.Zero}, {gainAt}
            global_load_dword {dgain}, {w.Zero}, {dgainAt}
            """);
        w.Columns(cols, e => $"global_load_dword {vs[e.K]}, {e.Off}, {vb}{e.Imm}\nglobal_load_dword {gs[e.K]}, {e.Off}, {gb}{e.Imm}", e => $"""
            v_fma_f32 {ss}, {vs[e.K]}, {vs[e.K]}, {ss}
            v_fma_f32 {dot}, {gs[e.K]}, {vs[e.K]}, {dot}
            """);
        w.Reduce(RowWriter.Add, ss, dot);
        RowMath(w, pw =>
        {
            pw.Sqrt(root, ss);
            pw.Div(inv, "1.0", root);
            pw.I(TrainingText($"v_mul_f32_e32 {scale}, {gain}, {inv}"));
            pw.I(TrainingText($"v_mul_f32_e32 {t}, {scale}, {dot}"));
            pw.Div(along, t, ss);
        });
        w.L($"""
            v_xor_b32_e32 {along}, 0x80000000, {along}
            v_mul_f32_e32 {t}, {dot}, {inv}
            v_add_f32_e32 {dgain}, {dgain}, {t}
            s_bitcmp1_b32 {flags}, 0
            s_cbranch_scc0 .Lno_gain
            """);
        w.StoreFirstLane((dgainAt, dgain));
        w.Label(".Lno_gain");
        w.L($"""
            s_bitcmp1_b32 {flags}, 1
            s_cbranch_scc0 .Lno_direction
            """);
        w.Columns(cols,
            e => $"""
                global_load_dword {vs[e.K]}, {e.Off}, {vb}{e.Imm}
                global_load_dword {gs[e.K]}, {e.Off}, {gb}{e.Imm}
                global_load_dword {ds[e.K]}, {e.Off}, {db}{e.Imm}
                """,
            e => $"""
                v_fma_f32 {ts[e.K]}, {gs[e.K]}, {scale}, {ds[e.K]}
                v_fma_f32 {ts[e.K]}, {vs[e.K]}, {along}, {ts[e.K]}
                global_store_dword {e.Off}, {ts[e.K]}, {db}{e.Imm}
                """);
        w.Label(".Lno_direction");
        w.EndRows();
        return w.Source();
    }
}

// The attention backward passes in GCN (plan 35, G5b; moved from G4b), the gradients of HIP's attention_decode_f32 used
// as the tiled form (a key/value cache [heads, cap, dim], row i of head h seeing positions [begin, end) with end =
// min(i mod steps, cap - 1) + 1, begin = end - window) and of attention_spans_f32 (keys and values [kv heads, keyrows,
// dim], query head h reading key/value head h / group, row i its range from starts and ends[(h / hpt) · rows + i]). The
// attention weights are recomputed, never stored: P = e^(s - lse) with s the scaled (and soft-capped) score, dS = P ·
// (dout · v - Δ) · the cap's slope, Δ = dout · output per row (attn_bwd_d_f32).
// - attn_bwd_q_f32 / attn_spans_bwd_q_f32: one block a query row (grid-stride); the query and dout rows go to LDS once;
//   tiles of 256 keys, a thread a key (its two dot products over the head, four dimensions a step), the tile's scale · dS
//   to LDS, then each thread adds scale · dS · key over the tile for its dimensions (d = tid + 256 k), in key order.
// - attn_bwd_kv_f32 / attn_spans_bwd_kv_f32: one block a key position of a key/value head (grid-stride); its key and
//   value rows go to LDS once; for each query head of its group, in order, tiles of 256 query rows, a thread a row (the
//   rows that do not see the key give 0), the tile's P and scale · dS to LDS, then each thread adds scale · dS · q and
//   P · dout over the tile for its dimensions, in row order.
// So every gradient is summed in an order fixed by the shapes, owned by one block: no atomics, the same bits on every grid
// and device (rule 86). Head sizes up to AttentionMaxDim, byte offsets within a head 32-bit (the backend checks).
internal static partial class GcnKernels
{
    // The arguments: CUDA's (PtxKernels.Rows.cs attn_bwd_*, .SpansBackward.cs), then the spans' head count and n.
    private static readonly string[] TiledBackwardPointers = ["q", "keys", "values", "dout", "lse", "delta", "dq", "dk", "dv"];

    private static readonly string[] TiledBackwardScalars = ["rph", "steps", "cap", "dim", "scale", "window", "softcap", "n"];

    private static readonly string[] SpansBackwardPointers = ["q", "keys", "values", "starts", "ends", "dout", "lse", "delta", "dq", "dk", "dv"];

    private static readonly string[] SpansBackwardScalars = ["rows", "keyrows", "dim", "scale", "group", "hpt", "softcap", "heads", "n"];

    /// <summary>The LDS an attention backward block declares: two head rows and two tiles of weights.</summary>
    internal const int AttentionBackwardLdsBytes = 8 * AttentionMaxDim + 8 * AttentionTile;

    private static IEnumerable<GcnKernelSource> TrainingAttention()
    {
        yield return AttentionDelta();
        yield return AttentionBackwardQueries(spans: false);
        yield return AttentionBackwardKeys(spans: false);
        yield return AttentionBackwardQueries(spans: true);
        yield return AttentionBackwardKeys(spans: true);
    }

    // Δ[row] = dout[row] · o[row] over dim, one block a row.
    private static GcnKernelSource AttentionDelta()
    {
        var w = new RowWriter("attn_bwd_d_f32", arrays: 1);
        var a = w.Arguments(["o", "dout", "dd"], ["dim", "n"]);
        string dim = a.Scalar("dim");
        w.BeginRows(a.Scalar("n"));
        int offset = w.RowOffset(dim);
        string ob = w.Base(a.Pointer("o"), offset), gb = w.Base(a.Pointer("dout"), offset);
        var os = w.Vs();
        var gs = w.Vs();
        string dot = w.V();
        w.L($"v_mov_b32_e32 {dot}, 0");
        w.Columns(dim, e => $"global_load_dword {os[e.K]}, {e.Off}, {ob}{e.Imm}\nglobal_load_dword {gs[e.K]}, {e.Off}, {gb}{e.Imm}",
            e => $"v_fma_f32 {dot}, {gs[e.K]}, {os[e.K]}, {dot}");
        w.Reduce(RowWriter.Add, dot);
        w.StoreFirstLane((w.ElementAddress(a.Pointer("dd"), w.Row), dot));
        w.EndRows();
        return w.Source();
    }

    // The dimensions a thread holds (d = tid + 256 k) and their byte offsets.
    private static (string[] Dims, string[] Bytes) AttentionSlotsOf(RowWriter w)
    {
        var dims = new string[AttentionSlots];
        var bytes = new string[AttentionSlots];
        for (int k = 0; k < AttentionSlots; k++)
        {
            (dims[k], bytes[k]) = (w.V(), w.V());
            w.L($"v_add_u32_e32 {dims[k]}, 0x{RowThreads * k:x}, {w.Tid}");
            w.L($"v_lshlrev_b32_e32 {bytes[k]}, 2, {dims[k]}");
        }

        return (dims, bytes);
    }

    // Each slot's element of two rows (at a and b) into LDS at ldsA and ldsB, then a barrier.
    private static void AttentionRowsToLds(RowWriter w, string label, string dim, string[] dims, string[] bytes, string a, string b, int ldsA, int ldsB)
    {
        for (int k = 0; k < AttentionSlots; k++)
        {
            string saved = w.SPair(), va = w.V(), vb = w.V();
            if (k > 0)
            {
                w.L($"s_cmp_le_u32 {dim}, 0x{RowThreads * k:x}");
                w.L($"s_cbranch_scc1 {label}{k}");
            }

            w.L($"""
                v_cmp_gt_u32_e32 vcc, {dim}, {dims[k]}
                s_and_saveexec_b64 {saved}, vcc
                global_load_dword {va}, {bytes[k]}, {a}
                global_load_dword {vb}, {bytes[k]}, {b}
                ds_write_b32 {bytes[k]}, {va} offset:{ldsA}
                ds_write_b32 {bytes[k]}, {vb} offset:{ldsB}
                s_or_b64 exec, exec, {saved}
                """);
            if (k > 0)
            {
                w.Label($"{label}{k}");
            }
        }

        w.L($"s_barrier");
    }

    // Two dot products over the head in dimension order, four a step: Σ x[d] · ldsX[d] and Σ y[d] · ldsY[d], x and y read
    // at `at` (a byte offset, stepped) from bases xb and yb; into dotX and dotY.
    private static void AttentionDots(RowWriter w, string label, string dim, string at, string xb, string yb, int ldsX, int ldsY, string dotX, string dotY)
    {
        string readX = w.V(), d = w.S(), dim4 = w.S();
        w.L($"""
            v_mov_b32_e32 {dotX}, 0
            v_mov_b32_e32 {dotY}, 0
            v_mov_b32_e32 {readX}, 0
            s_mov_b32 {d}, 0
            s_and_b32 {dim4}, {dim}, -4
            """);
        w.Label(label);
        w.L($"s_cmp_ge_u32 {d}, {dim4}");
        w.L($"s_cbranch_scc1 {label}_end");
        string lx = AttentionText($"%v[{AttentionRange(w, 4)}]"), ly = AttentionText($"%v[{AttentionRange(w, 4)}]");
        int fx = AttentionFirst(lx), fy = AttentionFirst(ly);
        var xs = new string[4];
        var ys = new string[4];
        for (int e = 0; e < 4; e++)
        {
            (xs[e], ys[e]) = (w.V(), w.V());
            w.L($"global_load_dword {xs[e]}, {at}, {xb}{AttentionOffset(4 * e)}");
            w.L($"global_load_dword {ys[e]}, {at}, {yb}{AttentionOffset(4 * e)}");
        }

        w.L($"ds_read_b128 {lx}, {readX} offset:{ldsX}");
        w.L($"ds_read_b128 {ly}, {readX} offset:{ldsY}");
        for (int e = 0; e < 4; e++)
        {
            w.L($"v_fma_f32 {dotX}, %v{fx + e}, {xs[e]}, {dotX}");
            w.L($"v_fma_f32 {dotY}, %v{fy + e}, {ys[e]}, {dotY}");
        }

        w.L($"""
            v_add_u32_e32 {at}, 16, {at}
            v_add_u32_e32 {readX}, 16, {readX}
            s_add_u32 {d}, {d}, 4
            s_branch {label}
            """);
        w.Label(label + "_end");
        string rest = w.S();
        w.L($"s_and_b32 {rest}, {dim}, 3");
        for (int e = 0; e < 3; e++)
        {
            string x = w.V(), y = w.V(), lxe = w.V(), lye = w.V();
            w.L($"""
                s_cmp_le_u32 {rest}, {e}
                s_cbranch_scc1 {label}_tail
                global_load_dword {x}, {at}, {xb}{AttentionOffset(4 * e)}
                global_load_dword {y}, {at}, {yb}{AttentionOffset(4 * e)}
                ds_read_b32 {lxe}, {readX} offset:{ldsX + 4 * e}
                ds_read_b32 {lye}, {readX} offset:{ldsY + 4 * e}
                v_fma_f32 {dotX}, {lxe}, {x}, {dotX}
                v_fma_f32 {dotY}, {lye}, {y}, {dotY}
                """);
        }

        w.Label(label + "_tail");
    }

    // From the dot products: P = e^(s - lse) (s the scaled score, soft-capped as the forward pass caps it), and scale · dS
    // with dS = P · (dp - Δ) · the cap's slope 1 - (capped / softcap)².
    private static (string P, string Ds) AttentionGradient(RowWriter w, string label, string dot, string dp, string lse, string delta, string scale, string softcap,
        string capInverse)
    {
        string score = w.V(), slope = w.V(), c = w.V(), t = w.V(), p = w.V(), ds = w.V();
        w.L($"""
            v_mul_f32_e32 {score}, {scale}, {dot}
            v_mov_b32_e32 {slope}, 1.0
            s_cmp_gt_i32 {softcap}, 0
            s_cbranch_scc0 {label}
            v_mul_f32_e32 {c}, {score}, {capInverse}
            v_mul_f32_e32 {c}, 0x4038aa3b, {c}
            v_exp_f32_e32 {c}, {c}
            v_add_f32_e32 {c}, 1.0, {c}
            v_rcp_f32_e32 {c}, {c}
            v_fma_f32 {c}, {c}, -2.0, 1.0
            v_mul_f32_e32 {score}, {softcap}, {c}
            v_mul_f32_e32 {t}, {c}, {c}
            v_sub_f32_e32 {slope}, 1.0, {t}
            """);
        w.Label(label);
        w.L($"""
            {RowWriter.Exp(p, score, lse)}
            v_sub_f32_e32 {t}, {dp}, {delta}
            v_mul_f32_e32 {ds}, {p}, {t}
            v_mul_f32_e32 {ds}, {ds}, {slope}
            v_mul_f32_e32 {ds}, {scale}, {ds}
            """);
        return (p, ds);
    }

    // For each slot, acc[k] += Σ_j weight[j] · x[(first + j) · rowBytes + d] over `count` rows from LDS weights at
    // `weights` (and acc2 with the second weights and rows, when given), in j order, four rows a step.
    private static void AttentionAccumulate(RowWriter w, string label, string dim, string count, string firstBytes, string rowBytes, string[] dims, string[] bytes,
        (string Base, int Weights, string[] Acc)[] sums)
    {
        for (int k = 0; k < AttentionSlots; k++)
        {
            string saved = w.SPair(), o = w.V(), read = w.V(), j = w.S(), count4 = w.S();
            if (k > 0)
            {
                w.L($"s_cmp_le_u32 {dim}, 0x{RowThreads * k:x}");
                w.L($"s_cbranch_scc1 {label}{k}_skip");
            }

            w.L($"""
                v_cmp_gt_u32_e32 vcc, {dim}, {dims[k]}
                s_and_saveexec_b64 {saved}, vcc
                v_add_u32_e32 {o}, {firstBytes}, {bytes[k]}
                v_mov_b32_e32 {read}, 0
                s_mov_b32 {j}, 0
                s_and_b32 {count4}, {count}, -4
                """);
            w.Label($"{label}{k}");
            w.L($"s_cmp_ge_u32 {j}, {count4}");
            w.L($"s_cbranch_scc1 {label}{k}_tail");
            var at = new string[4];
            at[0] = o;
            for (int e = 1; e < 4; e++)
            {
                at[e] = w.V();
                w.L($"v_add_u32_e32 {at[e]}, {rowBytes}, {at[e - 1]}");
            }

            foreach (var (rowsBase, weights, acc) in sums)
            {
                var vs = new string[4];
                for (int e = 0; e < 4; e++)
                {
                    vs[e] = w.V();
                    w.L($"global_load_dword {vs[e]}, {at[e]}, {rowsBase}");
                }

                string ws = AttentionText($"%v[{AttentionRange(w, 4)}]");
                int wf = AttentionFirst(ws);
                w.L($"ds_read_b128 {ws}, {read} offset:{weights}");
                for (int e = 0; e < 4; e++)
                {
                    w.L($"v_fma_f32 {acc[k]}, %v{wf + e}, {vs[e]}, {acc[k]}");
                }
            }

            w.L($"""
                v_add_u32_e32 {o}, {rowBytes}, {at[3]}
                v_add_u32_e32 {read}, 16, {read}
                s_add_u32 {j}, {j}, 4
                s_branch {label}{k}
                """);
            w.Label($"{label}{k}_tail");
            w.L($"s_cmp_ge_u32 {j}, {count}");
            w.L($"s_cbranch_scc1 {label}{k}_end");
            foreach (var (rowsBase, weights, acc) in sums)
            {
                string v1 = w.V(), w1 = w.V();
                w.L($"""
                    global_load_dword {v1}, {o}, {rowsBase}
                    ds_read_b32 {w1}, {read} offset:{weights}
                    v_fma_f32 {acc[k]}, {w1}, {v1}, {acc[k]}
                    """);
            }

            w.L($"""
                v_add_u32_e32 {o}, {rowBytes}, {o}
                v_add_u32_e32 {read}, 4, {read}
                s_add_u32 {j}, {j}, 1
                s_branch {label}{k}_tail
                """);
            w.Label($"{label}{k}_end");
            w.L($"s_or_b64 exec, exec, {saved}");
            if (k > 0)
            {
                w.Label($"{label}{k}_skip");
            }
        }
    }

    // Each slot's accumulators added into a row at `target`.
    private static void AttentionAddRow(RowWriter w, string label, string dim, string[] dims, string[] bytes, string target, string[] acc)
    {
        for (int k = 0; k < AttentionSlots; k++)
        {
            string saved = w.SPair(), old = w.V();
            if (k > 0)
            {
                w.L($"s_cmp_le_u32 {dim}, 0x{RowThreads * k:x}");
                w.L($"s_cbranch_scc1 {label}{k}");
            }

            w.L($"""
                v_cmp_gt_u32_e32 vcc, {dim}, {dims[k]}
                s_and_saveexec_b64 {saved}, vcc
                global_load_dword {old}, {bytes[k]}, {target}
                v_add_f32_e32 {old}, {old}, {acc[k]}
                global_store_dword {bytes[k]}, {old}, {target}
                s_or_b64 exec, exec, {saved}
                """);
            if (k > 0)
            {
                w.Label($"{label}{k}");
            }
        }
    }

    private static string[] AttentionZeros(RowWriter w)
    {
        var acc = new string[AttentionSlots];
        for (int k = 0; k < AttentionSlots; k++)
        {
            acc[k] = w.V();
            w.L($"v_mov_b32_e32 {acc[k]}, 0");
        }

        return acc;
    }

    // dq of one query row a block (see the remarks above).
    private static GcnKernelSource AttentionBackwardQueries(bool spans)
    {
        var w = new RowWriter(spans ? "attn_spans_bwd_q_f32" : "attn_bwd_q_f32", arrays: 0, extraLds: AttentionBackwardLdsBytes);
        int queryLds = w.ExtraLdsBase, doutLds = queryLds + 4 * AttentionMaxDim, weightLds = doutLds + 4 * AttentionMaxDim;
        var a = spans ? w.Arguments(SpansBackwardPointers, SpansBackwardScalars) : w.Arguments(TiledBackwardPointers, TiledBackwardScalars);
        string dim = a.Scalar("dim"), scale = a.Scalar("scale"), softcap = a.Scalar("softcap");
        string rowBytes = w.S(), capInverse = w.V(), tid4 = w.V();
        w.L($"""
            s_lshl_b32 {rowBytes}, {dim}, 2
            v_rcp_f32_e32 {capInverse}, {softcap}
            v_lshlrev_b32_e32 {tid4}, 2, {w.Tid}
            """);
        var (dims, bytes) = AttentionSlotsOf(w);
        string byRph = "", bySteps = "", last = "", capRow = "", byRows = "", byGroup = "", byTables = "", kvRow = "";
        if (spans)
        {
            (byRows, byGroup, byTables) = (AttentionReciprocal(w, a.Scalar("rows")), AttentionReciprocal(w, a.Scalar("group")), AttentionReciprocal(w, a.Scalar("hpt")));
            kvRow = w.S();
            w.L($"s_mul_i32 {kvRow}, {a.Scalar("keyrows")}, {dim}");
        }
        else
        {
            (last, capRow) = (w.S(), w.S());
            w.L($"s_sub_u32 {last}, {a.Scalar("cap")}, 1");
            w.L($"s_mul_i32 {capRow}, {a.Scalar("cap")}, {dim}");
            (byRph, bySteps) = (AttentionReciprocal(w, a.Scalar("rph")), AttentionReciprocal(w, a.Scalar("steps")));
        }

        w.BeginRows(a.Scalar("n"));
        string begin = w.S(), end = w.S(), kvOffset;
        if (spans)
        {
            var (h, i) = AttentionDivMod(w, w.Row, a.Scalar("rows"), byRows);
            var (g, _) = AttentionDivMod(w, h, a.Scalar("group"), byGroup);
            var (table, _) = AttentionDivMod(w, h, a.Scalar("hpt"), byTables);
            AttentionSpanOf(w, a, table, i, begin, end);
            kvOffset = AttentionIndex(w, (g, kvRow));
        }
        else
        {
            var (head, index) = AttentionDivMod(w, w.Row, a.Scalar("rph"), byRph);
            var (_, step) = AttentionDivMod(w, index, a.Scalar("steps"), bySteps);
            string window = a.Scalar("window"), t = w.S();
            w.L($"""
                s_min_u32 {end}, {step}, {last}
                s_add_u32 {end}, {end}, 1
                s_sub_u32 {t}, {end}, {window}
                s_cmp_gt_u32 {end}, {window}
                s_cselect_b32 {begin}, {t}, 0
                s_cmp_eq_u32 {window}, 0
                s_cselect_b32 {begin}, 0, {begin}
                """);
            kvOffset = AttentionIndex(w, (head, capRow));
        }

        string queryOffset = AttentionIndex(w, (w.Row, dim));
        string qb = w.ElementAddress(a.Pointer("q"), queryOffset), ob = w.ElementAddress(a.Pointer("dout"), queryOffset), dqb = w.ElementAddress(a.Pointer("dq"), queryOffset);
        string kb = w.ElementAddress(a.Pointer("keys"), kvOffset), vb = w.ElementAddress(a.Pointer("values"), kvOffset);
        string lse = w.V(), delta = w.V();
        w.L($"global_load_dword {lse}, {w.Zero}, {w.ElementAddress(a.Pointer("lse"), w.Row)}");
        w.L($"global_load_dword {delta}, {w.Zero}, {w.ElementAddress(a.Pointer("delta"), w.Row)}");
        AttentionRowsToLds(w, ".Lq_rows", dim, dims, bytes, qb, ob, queryLds, doutLds);
        var acc = AttentionZeros(w);
        string c0 = w.S(), count = w.S(), inTile = w.SPair(), key = w.V(), at = w.V(), dot = w.V(), dp = w.V(), weight = w.V(), tileBytes = w.S();
        w.L($"s_mov_b32 {c0}, {begin}");
        w.Label(".Lq_tile");
        w.L($"""
            s_cmp_ge_u32 {c0}, {end}
            s_cbranch_scc1 .Lq_tiles_end
            s_sub_u32 {count}, {end}, {c0}
            s_min_u32 {count}, {count}, 0x{AttentionTile:x}
            v_mov_b32_e32 {weight}, 0
            v_cmp_gt_u32_e32 vcc, {count}, {w.Tid}
            s_and_saveexec_b64 {inTile}, vcc
            v_add_u32_e32 {key}, {c0}, {w.Tid}
            v_mul_lo_u32 {at}, {key}, {rowBytes}
            """);
        AttentionDots(w, ".Lq_dot", dim, at, kb, vb, queryLds, doutLds, dot, dp);
        var (_, ds) = AttentionGradient(w, ".Lq_capped", dot, dp, lse, delta, scale, softcap, capInverse);
        w.L($"""
            v_mov_b32_e32 {weight}, {ds}
            s_or_b64 exec, exec, {inTile}
            ds_write_b32 {tid4}, {weight} offset:{weightLds}
            s_barrier
            s_mul_i32 {tileBytes}, {c0}, {rowBytes}
            """);
        AttentionAccumulate(w, ".Lq_acc", dim, count, tileBytes, rowBytes, dims, bytes, [(kb, weightLds, acc)]);
        w.L($"""
            s_barrier
            s_add_u32 {c0}, {c0}, 0x{AttentionTile:x}
            s_branch .Lq_tile
            """);
        w.Label(".Lq_tiles_end");
        AttentionAddRow(w, ".Lq_out", dim, dims, bytes, dqb, acc);
        w.L($"s_barrier");
        w.EndRows();
        return w.Source();
    }

    // A key-range row's [begin, end) from starts and ends[table · rows + i] (floats), clamped to [0, keyrows].
    private static void AttentionSpanOf(RowWriter w, RowWriter.Args a, string table, string i, string begin, string end)
    {
        string at = AttentionIndex(w, (table, a.Scalar("rows")));
        w.L($"s_add_u32 {at}, {at}, {i}");
        string startAt = w.ElementAddress(a.Pointer("starts"), at), endAt = w.ElementAddress(a.Pointer("ends"), at);
        string first = w.V(), past = w.V(), keyRows = a.Scalar("keyrows");
        w.L($"""
            global_load_dword {first}, {w.Zero}, {startAt}
            global_load_dword {past}, {w.Zero}, {endAt}
            v_cvt_i32_f32_e32 {first}, {first}
            v_cvt_i32_f32_e32 {past}, {past}
            v_readfirstlane_b32 {begin}, {first}
            v_readfirstlane_b32 {end}, {past}
            s_max_i32 {begin}, {begin}, 0
            s_min_i32 {begin}, {begin}, {keyRows}
            s_max_i32 {end}, {end}, 0
            s_min_i32 {end}, {end}, {keyRows}
            """);
    }

    // dk and dv of one key position a block (see the remarks above).
    private static GcnKernelSource AttentionBackwardKeys(bool spans)
    {
        var w = new RowWriter(spans ? "attn_spans_bwd_kv_f32" : "attn_bwd_kv_f32", arrays: 0, extraLds: AttentionBackwardLdsBytes);
        int keyLds = w.ExtraLdsBase, valueLds = keyLds + 4 * AttentionMaxDim, pLds = valueLds + 4 * AttentionMaxDim, dsLds = pLds + 4 * AttentionTile;
        var a = spans ? w.Arguments(SpansBackwardPointers, SpansBackwardScalars) : w.Arguments(TiledBackwardPointers, TiledBackwardScalars);
        string dim = a.Scalar("dim"), scale = a.Scalar("scale"), softcap = a.Scalar("softcap");
        string positions = a.Scalar(spans ? "keyrows" : "cap"), rows = a.Scalar(spans ? "rows" : "rph");
        string rowBytes = w.S(), capInverse = w.V(), tid4 = w.V(), last = w.S();
        w.L($"""
            s_lshl_b32 {rowBytes}, {dim}, 2
            v_rcp_f32_e32 {capInverse}, {softcap}
            v_lshlrev_b32_e32 {tid4}, 2, {w.Tid}
            s_sub_u32 {last}, {positions}, 1
            """);
        var (dims, bytes) = AttentionSlotsOf(w);
        string byPositions = AttentionReciprocal(w, positions);
        string bySteps = spans ? "" : AttentionReciprocal(w, a.Scalar("steps"));
        string byTables = spans ? AttentionReciprocal(w, a.Scalar("hpt")) : "";
        w.BeginRows(a.Scalar("n"));
        var (g, c) = AttentionDivMod(w, w.Row, positions, byPositions);
        string kvOffset = AttentionIndex(w, (w.Row, dim));
        string kb = w.ElementAddress(a.Pointer("keys"), kvOffset), vb = w.ElementAddress(a.Pointer("values"), kvOffset);
        string dkb = w.ElementAddress(a.Pointer("dk"), kvOffset), dvb = w.ElementAddress(a.Pointer("dv"), kvOffset);
        AttentionRowsToLds(w, ".Lkv_rows", dim, dims, bytes, kb, vb, keyLds, valueLds);
        var accK = AttentionZeros(w);
        var accV = AttentionZeros(w);
        string h = w.S(), hEnd = w.S();
        if (spans)
        {
            w.L($"s_mul_i32 {h}, {g}, {a.Scalar("group")}");
            w.L($"s_add_u32 {hEnd}, {h}, {a.Scalar("group")}");
        }
        else
        {
            w.L($"s_mov_b32 {h}, {g}");
            w.L($"s_add_u32 {hEnd}, {g}, 1");
        }

        w.Label(".Lkv_head");
        w.L($"s_cmp_ge_u32 {h}, {hEnd}");
        w.L($"s_cbranch_scc1 .Lkv_heads_end");
        string tableRow = "";
        if (spans)
        {
            var (table, _) = AttentionDivMod(w, h, a.Scalar("hpt"), byTables);
            tableRow = AttentionIndex(w, (table, rows));
        }

        string headRow = AttentionIndex(w, (h, rows)), headOffset = AttentionIndex(w, (headRow, dim));
        string qb = w.ElementAddress(a.Pointer("q"), headOffset), ob = w.ElementAddress(a.Pointer("dout"), headOffset);
        string lseb = w.ElementAddress(a.Pointer("lse"), headRow), deltab = w.ElementAddress(a.Pointer("delta"), headRow);
        string r0 = w.S(), count = w.S(), inTile = w.SPair(), i = w.V(), sees = w.SPair(), first = w.V(), past = w.V(), t = w.V();
        string pw = w.V(), dw = w.V(), at = w.V(), rowAt = w.V(), lse = w.V(), delta = w.V(), dot = w.V(), dp = w.V(), tileBytes = w.S();
        w.L($"s_mov_b32 {r0}, 0");
        w.Label(".Lkv_tile");
        w.L($"""
            s_cmp_ge_u32 {r0}, {rows}
            s_cbranch_scc1 .Lkv_tiles_end
            s_sub_u32 {count}, {rows}, {r0}
            s_min_u32 {count}, {count}, 0x{AttentionTile:x}
            v_mov_b32_e32 {pw}, 0
            v_mov_b32_e32 {dw}, 0
            v_cmp_gt_u32_e32 vcc, {count}, {w.Tid}
            s_and_saveexec_b64 {inTile}, vcc
            v_add_u32_e32 {i}, {r0}, {w.Tid}
            """);
        if (spans)
        {
            w.L($"""
                v_add_u32_e32 {at}, {tableRow}, {i}
                v_lshlrev_b32_e32 {at}, 2, {at}
                global_load_dword {first}, {at}, {a.Pointer("starts")}
                global_load_dword {past}, {at}, {a.Pointer("ends")}
                v_cvt_i32_f32_e32 {first}, {first}
                v_cvt_i32_f32_e32 {past}, {past}
                v_max_i32_e32 {first}, 0, {first}
                v_min_i32_e32 {first}, {positions}, {first}
                v_max_i32_e32 {past}, 0, {past}
                v_min_i32_e32 {past}, {positions}, {past}
                """);
        }
        else
        {
            var (_, step) = RowDivMod(w, i, a.Scalar("steps"), bySteps);
            string window = a.Scalar("window"), none = w.SPair();
            w.L($"""
                v_min_u32_e32 {past}, {last}, {step}
                v_add_u32_e32 {past}, 1, {past}
                v_subrev_u32_e32 {t}, {window}, {past}
                v_cmp_lt_u32_e32 vcc, {window}, {past}
                v_cndmask_b32_e32 {first}, 0, {t}, vcc
                s_cmp_eq_u32 {window}, 0
                s_cselect_b64 {none}, -1, 0
                v_cndmask_b32_e64 {first}, {first}, 0, {none}
                """);
        }

        string after = w.SPair();
        w.L($"""
            v_cmp_ge_u32_e64 {sees}, {c}, {first}
            v_cmp_lt_u32_e64 {after}, {c}, {past}
            s_and_b64 {sees}, {sees}, {after}
            s_and_b64 exec, exec, {sees}
            s_cbranch_execz .Lkv_unseen
            v_lshlrev_b32_e32 {t}, 2, {i}
            global_load_dword {lse}, {t}, {lseb}
            global_load_dword {delta}, {t}, {deltab}
            v_mul_lo_u32 {rowAt}, {i}, {rowBytes}
            """);
        AttentionDots(w, ".Lkv_dot", dim, rowAt, qb, ob, keyLds, valueLds, dot, dp);
        var (p, ds) = AttentionGradient(w, ".Lkv_capped", dot, dp, lse, delta, scale, softcap, capInverse);
        w.L($"""
            v_mov_b32_e32 {pw}, {p}
            v_mov_b32_e32 {dw}, {ds}
            """);
        w.Label(".Lkv_unseen");
        w.L($"""
            s_or_b64 exec, exec, {inTile}
            ds_write_b32 {tid4}, {pw} offset:{pLds}
            ds_write_b32 {tid4}, {dw} offset:{dsLds}
            s_barrier
            s_mul_i32 {tileBytes}, {r0}, {rowBytes}
            """);
        AttentionAccumulate(w, ".Lkv_acc", dim, count, tileBytes, rowBytes, dims, bytes, [(qb, dsLds, accK), (ob, pLds, accV)]);
        w.L($"""
            s_barrier
            s_add_u32 {r0}, {r0}, 0x{AttentionTile:x}
            s_branch .Lkv_tile
            """);
        w.Label(".Lkv_tiles_end");
        w.L($"""
            s_add_u32 {h}, {h}, 1
            s_branch .Lkv_head
            """);
        w.Label(".Lkv_heads_end");
        AttentionAddRow(w, ".Lkv_dk", dim, dims, bytes, dkb, accK);
        AttentionAddRow(w, ".Lkv_dv", dim, dims, bytes, dvb, accV);
        w.L($"s_barrier");
        w.EndRows();
        return w.Source();
    }
}
