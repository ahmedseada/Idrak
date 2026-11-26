// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;
using System.Runtime.CompilerServices;
using Idrak.Abstraction.Operations;
using static Idrak.Gpu.Cuda.CudaDriver;

namespace Idrak.Gpu.Cuda;

// Inference kernels: norms, rotary embeddings, attention over caches and over spans, gated activations.
internal sealed unsafe partial class CudaBackend : ISegmentedAttentionBackend, IRowAttentionBackend
{
    // Today's int8 kernel (the named route): the few-row kernels, or the plain kernel for more rows.
    private void Int8Plain(Storage x, Storage q, Storage scales, Storage y, int m, int n, int k)
    {
        int words = (n + 3) / 4;
        if (m > PtxKernels.GemvRows || k == 0)
        {
            Launch1D(K("int8_matmul_f32"), m * words, P(x), P(q), P(scales), P(y), U(k), U(words), U(n), U(m * words));
            return;
        }

        PackedFewRows((int)PackedFormat.Int8 * 4 + GemvPlain, x, q, scales, y, m, n, k, words);
    }

    // Today's bfloat16 product (the named route): more rows by the packed product, else expanded and the float32 product;
    // few rows by the few-row kernels.
    private void BFloat16Plain(Storage x, Storage packed, Storage y, int m, int n, int k)
    {
        int words = (n + 1) / 2;
        if (m > PtxKernels.GemvRows && PackedMatMulLargeNamed(PackedFormat.BFloat16, x, packed, null, y, m, n, k))
        {
            return;
        }

        if (m > PtxKernels.GemvRows || k == 0)
        {
            var w = Allocate(k * n, zeroed: false);
            try
            {
                BFloat16Dequantize(packed, w, k, n);
                MatMul(x, w, y, m, n, k, false, false, 0f);
            }
            finally
            {
                w.Release();
            }

            return;
        }

        PackedFewRows((int)PackedFormat.BFloat16 * 4 + GemvPlain, x, packed, packed, y, m, n, k, words);
    }

    /// <summary>Benchmarks only: the k splits of prompt-sized packed products on tensor cores instead of the heuristic's.</summary>
    internal static int? PackedSplits { get; set; }

    /// <summary>Benchmarks only: the number of blocks decoding attention splits the cache over instead of the heuristic's.</summary>
    internal static int? DecodeSplits { get; set; }

    /// <summary>Benchmarks only: the number of k splits of the few-row packed products instead of the heuristic's.</summary>
    internal static int? GemvSplits { get; set; }

    // Kernel names by format (0 int8, 1 int4, 2 bfloat16) and variant, built once: the per-token path never formats a name.
    // The norm-add-norm kernels follow the twelve (GemvNormAddNorm + format), so the variants before keep their tuning keys.
    internal const int GemvPlain = 0, GemvSilu = 1, GemvGelu = 2, GemvAddNorm = 3, GemvNormAddNorm = 12;
    private static readonly string[] GemvKernels =
    [
        "int8_gemv_f32", "int8_gemv_silu_f32", "int8_gemv_gelu_f32", "int8_gemv_addnorm_f32",
        "int4_gemv_f32", "int4_gemv_silu_f32", "int4_gemv_gelu_f32", "int4_gemv_addnorm_f32",
        "bf16_gemv_f32", "bf16_gemv_silu_f32", "bf16_gemv_gelu_f32", "bf16_gemv_addnorm_f32",
        "int8_gemv_normaddnorm_f32", "int4_gemv_normaddnorm_f32", "bf16_gemv_normaddnorm_f32",
    ];

    // The plain kernels compiled for fewer rows, [format * GemvRowVariants.Length + i] for GemvRowVariants[i] rows.
    private static readonly string[] GemvRowKernels =
    [
        "int8_gemv_r1_f32", "int8_gemv_r2_f32", "int8_gemv_r4_f32", "int4_gemv_r1_f32", "int4_gemv_r2_f32", "int4_gemv_r4_f32",
        "bf16_gemv_r1_f32", "bf16_gemv_r2_f32", "bf16_gemv_r4_f32",
    ];

    /// <summary>
    /// Tests and benchmarks: false runs the plain few-row products of 1 to 4 rows on the eight-row kernels instead of the
    /// ones compiled for fewer rows (the same results bit for bit, so a comparison of the two).
    /// </summary>
    internal static bool GemvRowKernelsOn { get; set; } = true;

    // The kernel of a few-row product: a plain product of m ≤ 4 rows on the kernel compiled for the fewest rows that hold
    // m (PtxKernels.GemvRowVariants: fewer accumulators and no predicated-off rows, so more blocks fit a multiprocessor
    // and more loads are in flight), equal bit for bit to the eight-row kernel; the fused kernels and more rows on the
    // eight-row kernels. No allocation (the per-token path).
    private static string GemvKernel(int variant, int m)
    {
        var counts = PtxKernels.GemvRowVariants;
        if (GemvRowKernelsOn && variant < GemvNormAddNorm && variant % 4 == GemvPlain && m >= 1)
        {
            for (int i = 0; i < counts.Length; i++)
            {
                if (m <= counts[i])
                {
                    return GemvRowKernels[variant / 4 * counts.Length + i];
                }
            }
        }

        return GemvKernels[variant];
    }

    private static readonly string[] GemvMultiKernels =
        ["int8_gemv_multi_f32", "int8_gemv_multi_act_f32", "int4_gemv_multi_f32", "int4_gemv_multi_act_f32", "bf16_gemv_multi_f32", "bf16_gemv_multi_act_f32"];

    // [format * 2 + (64-row tile ? 1 : 0)]
    private static readonly string[] PackedTensorKernels =
        ["gemm_tc_nn_int8w_f32", "gemm_tc_nn_int8w_m64_f32", "gemm_tc_nn_int4w_f32", "gemm_tc_nn_int4w_m64_f32", "gemm_tc_nn_bf16w_f32", "gemm_tc_nn_bf16w_m64_f32"];

    private static readonly string[] PackedMultiKernels =
    [
        "gemm_tc_nn_int8w_multi_f32", "gemm_tc_nn_int8w_multi_m64_f32", "gemm_tc_nn_int4w_multi_f32", "gemm_tc_nn_int4w_multi_m64_f32",
        "gemm_tc_nn_bf16w_multi_f32", "gemm_tc_nn_bf16w_multi_m64_f32",
    ];

    // [(bfloat16 ? 1 : 0) * 2 + (multi ? 1 : 0)]
    private static readonly string[] PackedLowRankKernels =
        ["gemm_tc_nn_int4w_lr_f32", "gemm_tc_nn_int4w_multi_lr_f32", "gemm_tc_nn_bf16w_lr_f32", "gemm_tc_nn_bf16w_multi_lr_f32"];

    // [format * 2 + (128 tile ? 1 : 0)]
    private static readonly string[] PackedGemmKernels =
        ["gemm64_int8_f32", "gemm128_int8_f32", "gemm64_int4_f32", "gemm128_int4_f32", "gemm64_bf16_f32", "gemm128_bf16_f32"];

    // k splits of a few-row packed product with `blocks` column blocks, before measuring (see CudaBackend.Tuning.cs):
    // about two blocks per SM, rounded to a power of two so the chunks stay multiples of the 64-row unrolled step.
    private int GemvSplitCount(int blocks, int k) => GemvSplitCount(_multiprocessors, blocks, k);

    internal static int GemvSplitCount(int multiprocessors, int blocks, int k)
    {
        int wanted = (2 * Math.Max(1, multiprocessors) + blocks - 1) / blocks;
        int splits = 1 << (int)Math.Round(Math.Log2(Math.Max(1, wanted)));
        return Math.Clamp(splits, 1, GemvMaxSplits(k));
    }

    private static int GemvMaxSplits(int k) => Math.Max(1, Math.Min(64, k / 64));

    // What a few-row product's split count is kept under: the kernel (format * 4 + GemvPlain / GemvSilu / GemvGelu /
    // GemvAddNorm, or GemvNormAddNorm + format) and the shape, so the fused kernels never take the plain kernel's choice.
    internal static TuneKey GemvSplitsKey(int variant, int m, int n, int k) => new(TuneOp.GemvSplits, variant, m, n, k);

    // Split counts the few-row products are measured with: 1, 2, 3, 4, 6, 8, 12, ... up to GemvMaxSplits (the kernels
    // take any chunk of k: a step of 64 rows, then single rows; --bench-gemv found 6 best on one card for the
    // products into 1024 columns, which the powers of two alone missed by 25%), without counts that give the same
    // chunks once rounded to `align` (int4: a 64-row group); of two such counts the power of two stays, so the
    // formula's choice (GemvSplitCount) is always one of them.
    internal static int[] GemvSplitCandidates(int k, int align)
    {
        Span<int> values = stackalloc int[SplitCountsLength];
        return values[..GemvSplitCandidates(k, align, values)].ToArray();
    }

    // The same into `values` (SplitCountsLength long), without allocating (the per-token path); returns how many.
    internal static int GemvSplitCandidates(int k, int align, Span<int> values)
    {
        Span<int> counts = stackalloc int[SplitCountsLength];
        Span<int> runs = stackalloc int[SplitCountsLength];               // splits that run, per value kept
        int found = 0;
        foreach (int wanted in counts[..SplitCounts(GemvMaxSplits(k), counts)])
        {
            int chunk = ((k + wanted - 1) / wanted + align - 1) / align * align;
            int count = (k + chunk - 1) / chunk;
            int at = runs[..found].IndexOf(count);
            if (at < 0)
            {
                runs[found] = count;
                values[found++] = wanted;
            }
            else if (int.IsPow2(wanted))
            {
                values[at] = wanted;
            }
        }

        return found;
    }

    // Few rows (decoding) through packed weights: read each weight word once, with enough blocks to keep every
    // multiprocessor busy; narrow matrices split k, and the last block of each column range adds the splits in order.
    // `align`: split boundaries fall on multiples of it (int4 splits start on a 64-row block).
    // `up`: the gated kernels' second input; `tail`: further arguments (the addnorm kernels').
    // `variant`: format * 4 + GemvPlain / GemvSilu / GemvGelu / GemvAddNorm, or GemvNormAddNorm + format (an index into GemvKernels).
    private void PackedFewRows(int variant, Storage x, Storage q, Storage scales, Storage y, int m, int n, int k, int words, int align = 1,
        Storage? up = null, ReadOnlySpan<ulong> tail = default)
    {
        int columnBlocks = (words + 31) / 32;
        var rows = new FewRows(GemvKernel(variant, m), x, q, scales, up, SplitCounters(columnBlocks + 1), m, n, k, words, align);
        int splits;
        if (GemvSplits is int forced)
        {
            splits = Math.Clamp(forced, 1, Math.Max(1, k / 16));
        }
        else
        {
            // Measured once per shape and kernel, timing the kernel that then runs. The fused add-and-normalize kernels
            // (`tail`: residual, sum, gain, normalized, eps, offset) are timed whole, the residual addition and the
            // normalization by the last block included, with their three outputs (y, sum, normalized) in scratch memory,
            // since the sum may be the residual itself. (Before, they took the plain kernel's choice for the shape, timed
            // without that last step: --bench-gemv showed 8 splits chosen where 16 was 15% faster on one card.)
            // Once known, the choice is read without allocating (no candidate array, no closure).
            int formula = GemvSplitCount(columnBlocks, k);
            Span<int> candidates = stackalloc int[SplitCountsLength];
            candidates = candidates[..GemvSplitCandidates(k, align, candidates)];
            var key = GemvSplitsKey(variant, m, n, k);
            if (!TryTuned(key, candidates, formula, out splits))
            {
                splits = MeasureFewRows(rows, key, candidates.ToArray(), formula, P(y), tail.IsEmpty ? null : tail.ToArray());
            }
        }

        RunFewRows(rows, P(y), tail, splits);
    }

    // A few-row packed product but for its output, further arguments and split count (PackedFewRows).
    private readonly record struct FewRows(string Kernel, Storage X, Storage Q, Storage Scales, Storage? Up, Storage Counters, int M, int N, int K,
        int Words, int Align)
    {
        public int ColumnBlocks => (Words + 31) / 32;
    }

    // The arguments of a few-row product: twelve, then `up` or the add-and-normalize kernels' six (nine with the norm of y).
    private const int FewRowsArguments = 21;

    // `measured`: a run of the split count's measurement, whose partial sums take a pooled block (the candidates' larger
    // splits are not kept); every other run's go to the kept block (no storage allocated a call).
    private void RunFewRows(in FewRows rows, ulong output, ReadOnlySpan<ulong> extra, int wanted, bool measured = false)
    {
        int k = rows.K, align = rows.Align;
        int chunk = ((k + wanted - 1) / wanted + align - 1) / align * align;
        int count = (k + chunk - 1) / chunk;
        var part = count <= 1 ? null : measured ? Allocate(count * rows.M * rows.N, zeroed: false) : _splitPartials.Get(this, count * rows.M * rows.N);
        try
        {
            Span<ulong> args = stackalloc ulong[FewRowsArguments];
            args[0] = P(rows.X);
            args[1] = P(rows.Q);
            args[2] = P(rows.Scales);
            args[3] = output;
            args[4] = part is null ? output : P(part);
            args[5] = U(rows.M);
            args[6] = U(rows.N);
            args[7] = U(k);
            args[8] = U(rows.Words);
            args[9] = U(chunk);
            args[10] = U(count);
            args[11] = P(rows.Counters);
            int used = 12;
            if (rows.Up is not null)
            {
                args[used++] = P(rows.Up);
            }

            extra.CopyTo(args[used..]);
            Launch(K(rows.Kernel), (uint)rows.ColumnBlocks, (uint)count, 1, PtxKernels.Int8GemvThreads, 1, args[..(used + extra.Length)]);
        }
        finally
        {
            if (measured)
            {
                part?.Release();
            }
        }
    }

    // Measures the split count of a few-row product: writing its output, or with `tail` (the add-and-normalize kernels)
    // their three outputs in scratch memory. The formula's choice when there is no room for the scratch.
    private int MeasureFewRows(FewRows rows, TuneKey key, int[] candidates, int formula, ulong output, ulong[]? tail)
    {
        if (tail is null)
        {
            return Tune(key, candidates, formula, c => RunFewRows(rows, output, [], c, measured: true), cold: true);
        }

        int splits = formula;
        long size = (long)rows.M * rows.N * sizeof(float);
        WithScratch(3L * rows.M * rows.N, scratch => splits = Tune(key, candidates, formula,
            c => RunFewRows(rows, scratch, [tail[0], scratch + (ulong)size, tail[2], scratch + 2 * (ulong)size, .. tail.AsSpan(4)], c, measured: true), cold: true));
        return splits;
    }

    /// <summary>
    /// Tests and benchmarks: with a value, 4-8 int8 rows go to the packed product from this many weights (k·n) on, without
    /// measuring; null (the default): measured per shape on this card.
    /// </summary>
    internal static long? PackedPreferredWeights { get; set; }

    private int MeasureInt8FewRows(TuneKey key, Storage x, Storage packed, Storage? scales, Storage y, int m, int n, int k)
    {
        void Run(int choice)
        {
            if (choice == 0 || !PackedMatMulLargeNamed(PackedFormat.Int8, x, packed, scales, y, m, n, k))
            {
                Int8Plain(x, packed, scales ?? packed, y, m, n, k);
            }
        }

        return Tune(key, [0, 1], 0, Run, cold: true);
    }

    /// <summary>Benchmarks only: the row tile (64 or 128) of prompt-sized packed products instead of the heuristic's.</summary>
    internal static int? PromptTileRowsOverride { get; set; }

    // Row tile of a prompt-sized packed product: 64 when the last 128-row tile would be at most half full (180 rows: 3
    // tiles of 64, 192 rows computed, instead of 2 of 128, 256), else 128.
    private static int PromptTileRows(int m) =>
        PromptTileRowsOverride ?? (m < 1024 && m % PtxKernels.TensorTile is > 0 and <= 64 ? 64 : PtxKernels.TensorTile);

    // k splits of a prompt-sized packed product before measuring (see PromptSplitsTuned): up to four blocks per SM, chunks
    // of 256 k or more; none when the tiles already fill about one wave of the SMs (splitting then only adds the partial
    // tiles' writes and their ordered sum).
    private int PromptSplits(int tiles, int k)
    {
        int sms = Math.Max(1, _multiprocessors);
        return tiles * 100 >= sms * 85 && tiles <= sms ? 1 : Math.Clamp(Math.Min(k / 256, 4 * sms / tiles), 1, 8);
    }

    // The k splits of a prompt-sized packed product, measured once per shape on this card: `run(splits)` launches, writing
    // outputs the final run writes again. With four or more waves of tiles there is nothing to gain from splitting, so
    // nothing is measured. Candidates stop at MaxSplits (the ordered sum's workspace: one partial tile per block).
    private int PromptSplitsTuned(TuneKey key, int tiles, int k, Action<int> run)
    {
        int formula = PromptSplits(tiles, k);
        if (tiles >= 4 * Math.Max(1, _multiprocessors))
        {
            return formula;
        }

        Span<int> candidates = stackalloc int[SplitCountsLength];
        return Tune(key, candidates[..SplitCounts(Math.Min(Math.Min(16, k / 128), MaxSplits(tiles)), candidates)], formula, run);
    }

    // PackedSplits' count (benchmarks) for a product of `tiles` output tiles: at least 1, chunks of 32 k or more, at most
    // MaxSplits.
    private int ForcedSplits(int forced, int k, int tiles) => Math.Clamp(forced, 1, Math.Max(1, Math.Min(k / 32, MaxSplits(tiles))));

    // [plain or residual] × [128-row tile, 64-row tile] for the epilogue products over bfloat16 weights; GELU apart.
    private static readonly string[] PackedEpilogueKernels = ["gemm_tc_nn_bf16w_f32", "gemm_tc_nn_bf16w_m64_f32", "gemm_tc_nn_bf16w_res_f32", "gemm_tc_nn_bf16w_m64_res_f32"];
    private static readonly string[] PackedGeluKernels = ["gemm_tc_nn_gelu_bf16w_f32", "gemm_tc_nn_gelu_bf16w_m64_f32"];

    // A multi-output few-row product but for its products and split count (PackedMany).
    private readonly record struct ManyRows(string Kernel, Storage X, Storage Counters, Storage? Hidden, int M, int K, int Nmax, int Align, int ColumnBlocks,
        int Activation);

    private int MeasureMany(ManyRows rows, TuneKey name, int[] candidates, int formula,
        (Storage Packed, Storage? Scales, Storage? Bias, Storage Output, int Columns)[] products) =>
        Tune(name, candidates, formula, c => RunMany(rows, products, c, measured: true), cold: true);

    // `measured`: as RunFewRows's.
    private void RunMany(in ManyRows rows, ReadOnlySpan<(Storage Packed, Storage? Scales, Storage? Bias, Storage Output, int Columns)> products, int wanted,
        bool measured = false)
    {
        int m = rows.M, k = rows.K, nmax = rows.Nmax, align = rows.Align, columnBlocks = rows.ColumnBlocks;
        var hidden = rows.Hidden;
        int chunk = ((k + wanted - 1) / wanted + align - 1) / align * align;
        int splits = (k + chunk - 1) / chunk;
        int partLength = products.Length * splits * m * nmax;
        var part = splits <= 1 ? null : measured ? Allocate(partLength, zeroed: false) : _splitPartials.Get(this, partLength);
        try
        {
            Span<ulong> args = stackalloc ulong[9 + 3 * 5 + 2];
            args[0] = P(rows.X);
            args[1] = part is null ? P(products[0].Output) : P(part);
            args[2] = U(m);
            args[3] = U(k);
            args[4] = U(chunk);
            args[5] = U(splits);
            args[6] = P(rows.Counters);
            args[7] = U(columnBlocks);
            args[8] = U(nmax);
            for (int j = 0; j < 3; j++)
            {
                var product = j < products.Length ? products[j] : products[0];
                args[9 + 5 * j] = P(product.Packed);
                args[10 + 5 * j] = P(product.Scales ?? product.Packed);
                args[11 + 5 * j] = P(product.Output);
                args[12 + 5 * j] = product.Bias is null ? 0UL : P(product.Bias);
                args[13 + 5 * j] = U(j < products.Length ? product.Columns : 0);
            }

            args[24] = U(rows.Activation);
            args[25] = hidden is null ? 0UL : P(hidden);
            Launch(K(rows.Kernel), (uint)columnBlocks, (uint)splits, (uint)products.Length, PtxKernels.Int8GemvThreads, 1, hidden is null ? args[..24] : args);
        }
        finally
        {
            if (measured)
            {
                part?.Release();
            }
        }
    }

    // The partial sums of split few-row products (RunFewRows, RunMany), kept between calls.
    private readonly KeptScratch _splitPartials = new();

    private Storage? _splitCounters;

    // Zeroed arrival counters for split products, one per column block; the last block of a range resets its counter.
    private Storage SplitCounters(int blocks)
    {
        // A larger buffer replaces a smaller one without freeing it: recorded graphs may still use the old one.
        var current = Volatile.Read(ref _splitCounters);
        if (current is null || current.Length < blocks)
        {
            current = Allocate(Math.Max(blocks, 4096), zeroed: true);
            Volatile.Write(ref _splitCounters, current);
        }

        return current;
    }

    public override void RmsNormKernel(Storage x, Storage y, Storage inv, int rows, int cols, float eps) =>
        LaunchRows(K("rms_norm_f32"), rows, P(x), P(y), P(inv), U(cols), F(eps), U(rows));

    public override void RmsNormBackwardKernel(Storage dy, Storage y, Storage inv, Storage dx, int rows, int cols) =>
        LaunchRows(K("rms_norm_backward_f32"), rows, P(dy), P(y), P(inv), P(dx), U(cols), U(rows));

    public override void RopeKernel(Storage x, Storage y, Storage cos, Storage sin, Storage positions, int rows, int heads, int steps, int dim, int half, bool interleaved, float sign)
    {
        int n = rows * half;
        Launch1D(K("rope_f32"), n, P(x), P(y), P(cos), P(sin), P(positions), U(heads), U(steps), U(dim), U(half), U(interleaved ? 1 : 0), F(sign), U(n));
    }

    public override void AttentionDecodeKernel(Storage q, Storage keys, Storage values, Storage position, Storage y, int heads, int rowsPerHead,
        int steps, int capacity, int dim, float scale, AttentionVariant variant = default)
    {
        if (dim > PtxKernels.DecodeMaxDim)
        {
            throw new NotSupportedException($"Decoding attention supports head sizes up to {PtxKernels.DecodeMaxDim} on CUDA.");
        }

        DecodeSplit(WindowedTuning(0, variant, capacity),
            new DecodeAttention(0, q, keys, values, null, null, y, rowsPerHead, steps, capacity, dim, scale, 0, variant, heads * rowsPerHead), position);
    }

    // The tuning variant of a decoding attention kernel (0 float32, 1 int8, 2 bfloat16): a window shorter than the
    // capacity is measured under its own (3 to 5), since its rows read only the window's positions.
    private static int WindowedTuning(int format, AttentionVariant variant, int capacity) =>
        variant.Window > 0 && variant.Window < capacity ? format + 3 : format;

    // Decoding attention has one block per query row (few rows: the heads of one token), so the cached positions are
    // split over several blocks per row; each writes (max, sum, weighted values) for its chunk and the last block of a
    // row to finish merges them (counted in the split counters). The split count depends only on the shapes, so
    // recorded graphs stay valid as the cache fills (chunks are computed on the device from the current length).
    // How many splits is measured once per shape (kernel `variant`, rows, capacity, head size) on this card, over
    // filled lengths 64, 128, ... up to the capacity (scratch positions), by the geometric mean of their times, so each
    // doubling of the context counts alike. (Measured at a full cache alone, one card chose 64 splits, 2%
    // faster there than 8, and ran 200 positions in 36.8 µs against 12.0 with 8.) Candidates: 1, 2, 3, 4, 6, 8, ... up to one split
    // per 64 cached positions (at most 64). The formula, about five blocks per SM (--bench-gemv: 16 rows → 22 splits,
    // 4000 positions 75.9 → ~35 µs, 1000 positions 22 → ~15 µs), is the default until then and when nothing can be
    // measured.
    // A short filled length split as many ways reads a few positions per block and merges many parts: on one card
    // 200 positions took 9.0 µs with the 24 splits measured at 4096, 7.6 with 16. So each block also takes at least
    // `minChunk` positions (the blocks past the filled length then add an empty part), measured per shape after the
    // splits over the same lengths; 1, the plain chunks, is the reference and stays unless another is faster. `attention`: the kernel's operands (RunDecode).
    // Rows and capacity are keyed by their size class (TuneSizes): a prompt's pieces and every conversation's own length
    // would otherwise measure the splits again at nearly every call; a class-mate's choice that is not one of this shape's
    // candidates gives the formula's (TryTuned).
    private void DecodeSplit(int variant, in DecodeAttention attention, Storage position)
    {
        int rows = attention.Rows, capacity = attention.Capacity, dim = attention.Dim;
        var counters = SplitCounters(rows);
        int limit = Math.Clamp(capacity / 64, 1, 64);
        int splits, minChunk = 1;
        if (DecodeSplits is int forced)
        {
            splits = Math.Clamp(forced, 1, 64);
            minChunk = DecodeMinChunk ?? 1;
        }
        else
        {
            // Once both are known, they are read without allocating (no candidate arrays, no closures).
            int formula = Math.Clamp((5 * Math.Max(1, _multiprocessors) + rows - 1) / rows, 1, Math.Min(32, limit));
            Span<int> candidates = stackalloc int[SplitCountsLength];
            candidates = candidates[..DecodeSplitCandidates(limit, formula, candidates)];
            Span<int> chunks = stackalloc int[DecodeChunksLength];
            bool known = TryTuned(new TuneKey(TuneOp.DecodeSplits, variant, TuneSizes.Class(rows), TuneSizes.Class(capacity), dim), candidates, formula, out splits);
            if (known && DecodeMinChunk is int forcedChunk)
            {
                minChunk = forcedChunk;
            }
            else if (known)
            {
                chunks = chunks[..(splits > 1 ? DecodeMinChunks(capacity, splits, chunks) : 0)];
                known = TryTuned(new TuneKey(TuneOp.DecodeMinChunk, variant, TuneSizes.Class(rows), TuneSizes.Class(capacity), dim, splits), chunks, 1, out minChunk);
            }

            if (!known)
            {
                (splits, minChunk) = MeasureDecode(variant, attention, counters, candidates.ToArray(), formula);
            }
        }

        RunDecode(attention, counters, splits, minChunk, P(position));
    }

    // Split counts decoding attention is measured with: SplitCounts up to `limit`, and the formula's choice in order.
    private static int DecodeSplitCandidates(int limit, int formula, Span<int> values)
    {
        int count = SplitCounts(limit, values);
        if (!values[..count].Contains(formula))
        {
            int at = count++;
            for (; at > 0 && values[at - 1] > formula; at--)
            {
                values[at] = values[at - 1];
            }

            values[at] = formula;
        }

        return count;
    }

    // Measures the split count, then the least chunk, over the filled lengths of DecodeTuneLengths (scratch positions).
    private (int Splits, int MinChunk) MeasureDecode(int variant, DecodeAttention attention, Storage counters, int[] candidates, int formula)
    {
        int rows = attention.Rows, capacity = attention.Capacity, dim = attention.Dim;
        int[] lengths = DecodeTuneLengths(capacity);
        Storage? positions = null;
        ulong At(int i)
        {
            if (positions is null)
            {
                positions = Allocate(lengths.Length, zeroed: false);
                Upload([.. lengths.Select(n => n - 1f)], positions);     // every row then reads that many positions
            }

            return P(positions) + (ulong)(4 * i);
        }

        try
        {
            int splits = Tune(new TuneKey(TuneOp.DecodeSplits, variant, TuneSizes.Class(rows), TuneSizes.Class(capacity), dim), candidates, formula, c =>
            {
                for (int i = 0; i < lengths.Length; i++)
                {
                    RunDecode(attention, counters, c, 1, At(i));             // writes y, which the launch after writes again
                }
            }, part: (c, i) => RunDecode(attention, counters, c, 1, At(i)), parts: lengths.Length);
            int[] chunks = splits > 1 ? DecodeMinChunks(capacity, splits) : [];
            int minChunk = DecodeMinChunk ?? Tune(new TuneKey(TuneOp.DecodeMinChunk, variant, TuneSizes.Class(rows), TuneSizes.Class(capacity), dim, splits), chunks, 1, c =>
            {
                for (int i = 0; i < lengths.Length; i++)
                {
                    RunDecode(attention, counters, splits, c, At(i));
                }
            }, part: (c, i) => RunDecode(attention, counters, splits, c, At(i)), parts: lengths.Length);
            return (splits, minChunk);
        }
        finally
        {
            positions?.Release();
        }
    }

    // A decoding attention launch but for its split count, least chunk, partial results, counters and position: the
    // cache format (0 float32, 1 int8, 2 bfloat16) and its operands (DecodeSplit).
    // Format 3: float32 rows of different lengths (attention_rows_f32), each from starts[(head / HeadsPerRow) · steps + step];
    // format 4: the same rows over a bfloat16 cache (attention_rows_bf16).
    private readonly record struct DecodeAttention(int Format, Storage Q, Storage Keys, Storage Values, Storage? KeyScales, Storage? ValueScales,
        Storage Y, int RowsPerHead, int Steps, int Capacity, int Dim, float Scale, int Words, AttentionVariant Variant, int Rows, Storage? Starts = null,
        int HeadsPerRow = 1);

    private void RunDecode(in DecodeAttention a, Storage counters, int splits, int minChunk, ulong at)
    {
        if (splits == 1)
        {
            LaunchDecode(a, 1, 1, a.Y, counters, at);
            return;
        }

        var part = Allocate(a.Rows * splits * (a.Dim + 2), zeroed: false);
        try
        {
            LaunchDecode(a, splits, minChunk, part, counters, at);
        }
        finally
        {
            part.Release();
        }
    }

    private void LaunchDecode(in DecodeAttention a, int splits, int minChunk, Storage part, Storage counters, ulong at)
    {
        switch (a.Format)
        {
            case 0:
                Launch(K("attention_decode_f32"), (uint)a.Rows, (uint)splits, 1, PtxKernels.RowThreads, 1,
                    P(a.Q), P(a.Keys), P(a.Values), at, P(a.Y), P(part), P(counters), U(a.RowsPerHead), U(a.Steps), U(a.Capacity), U(a.Dim), F(a.Scale), U(minChunk),
                    U(a.Variant.Window), F(a.Variant.Softcap), U(a.Rows));
                break;
            case 3:
                Launch(K("attention_rows_f32"), (uint)a.Rows, (uint)splits, 1, PtxKernels.RowThreads, 1,
                    P(a.Q), P(a.Keys), P(a.Values), at, P(a.Y), P(part), P(counters), P(a.Starts!), U(a.RowsPerHead), U(a.Steps), U(a.Capacity), U(a.Dim), F(a.Scale),
                    U(minChunk), U(a.Variant.Window), F(a.Variant.Softcap), U(a.HeadsPerRow), U(a.Rows));
                break;
            case 4:
                Launch(K("attention_rows_bf16"), (uint)a.Rows, (uint)splits, 1, PtxKernels.RowThreads, 1,
                    P(a.Q), P(a.Keys), P(a.Values), at, P(a.Y), P(part), P(counters), P(a.Starts!), U(a.RowsPerHead), U(a.Steps), U(a.Capacity), U(a.Dim), F(a.Scale),
                    U(a.Words), U(minChunk), U(a.Variant.Window), F(a.Variant.Softcap), U(a.HeadsPerRow), U(a.Rows));
                break;
            case 1:
                Launch(K("attention_decode_int8"), (uint)a.Rows, (uint)splits, 1, PtxKernels.RowThreads, 1,
                    P(a.Q), P(a.Keys), P(a.Values), P(a.KeyScales!), P(a.ValueScales!), at, P(a.Y), P(part), P(counters),
                    U(a.RowsPerHead), U(a.Steps), U(a.Capacity), U(a.Dim), F(a.Scale), U(a.Words), U(minChunk), U(a.Variant.Window), F(a.Variant.Softcap), U(a.Rows));
                break;
            default:
                Launch(K("attention_decode_bf16"), (uint)a.Rows, (uint)splits, 1, PtxKernels.RowThreads, 1,
                    P(a.Q), P(a.Keys), P(a.Values), at, P(a.Y), P(part), P(counters), U(a.RowsPerHead), U(a.Steps), U(a.Capacity), U(a.Dim), F(a.Scale), U(a.Words),
                    U(minChunk), U(a.Variant.Window), F(a.Variant.Softcap), U(a.Rows));
                break;
        }
    }

    /// <summary>Tests and benchmarks: the least positions per block of decoding attention, instead of the measured one.</summary>
    internal static int? DecodeMinChunk { get; set; }

    // Filled lengths the least chunk of decoding attention is timed at: 64, 128, ... below the capacity, then the capacity.
    internal static int[] DecodeTuneLengths(int capacity)
    {
        var lengths = new List<int>();
        for (int n = 64; n < capacity; n *= 2)
        {
            lengths.Add(n);
        }

        lengths.Add(Math.Max(1, capacity));
        return [.. lengths];
    }

    // Candidates for the least positions per block: 1 (plain chunks), then 8 (one per warp of a block) and its doublings
    // while a full cache still gives every split more than that (above, the full cache would run on fewer blocks than
    // measured best).
    internal static int[] DecodeMinChunks(int capacity, int splits)
    {
        Span<int> chunks = stackalloc int[DecodeChunksLength];
        return chunks[..DecodeMinChunks(capacity, splits, chunks)].ToArray();
    }

    // The same into `chunks` (DecodeChunksLength long), without allocating; returns how many.
    private static int DecodeMinChunks(int capacity, int splits, Span<int> chunks)
    {
        int count = 0;
        chunks[count++] = 1;
        for (int c = 8; c * splits < capacity; c *= 2)
        {
            chunks[count++] = c;
        }

        return count;
    }

    // 1, then 8 and its doublings below any int capacity: at most 29.
    private const int DecodeChunksLength = 32;

    public override void RmsNormAffineKernel(Storage x, Storage gain, Storage y, int rows, int cols, float eps, float offset) =>
        LaunchRows(K("rms_norm_affine_f32"), rows, P(x), P(gain), P(y), U(cols), F(eps), F(offset), U(rows));

    public override void GatedActivationKernel(Storage gate, Storage up, Storage y, int n, int kind) =>
        Launch1D(K("gated_act_f32"), n, P(gate), P(up), P(y), U(kind), U(n));

    public override void GatedActivationBackwardKernel(Storage gate, Storage up, Storage dy, Storage dgate, Storage dup, int n, int kind, int flags) =>
        Launch1D(K("gated_act_bwd_f32"), n, P(gate), P(up), P(dy), P(dgate), P(dup), U(kind), U(flags), U(n));

    // Windows and soft-caps run on the float32 kernels (attention_flash_f32 and its backward); the tensor-core ones
    // compute plain causal attention only.
    public override void AttentionTiledKernel(Storage q, Storage keys, Storage values, Storage position, Storage y, Storage? logSumExp, int heads,
        int rowsPerHead, int steps, int capacity, int dim, float scale, AttentionVariant variant = default)
    {
        if (variant.IsPlain && FlashTensorCore(dim, Ops.AttentionTiled, (long)heads * (rowsPerHead + 2L * steps) * dim) is { } tc)
        {
            Launch(tc[FlashNames(dim).Forward], (uint)((rowsPerHead + PtxKernels.FlashTensorRows - 1) / PtxKernels.FlashTensorRows), (uint)heads, 1, 128, 1,
                [P(q), P(keys), P(values), P(position), P(y), logSumExp is null ? 0UL : P(logSumExp),
                U(rowsPerHead), U(steps), U(capacity), F(scale * Log2E), .. CacheLayout(rowsPerHead, steps, capacity, dim, dim, variant.Merged)]);
            return;
        }

        NotMerged(variant, nameof(AttentionTiled));

        if (dim > PtxKernels.FlashMaxDim)
        {
            base.AttentionTiledKernel(q, keys, values, position, y, logSumExp, heads, rowsPerHead, steps, capacity, dim, scale, variant);
            return;
        }

        Launch(K("attention_flash_f32"), (uint)((rowsPerHead + PtxKernels.FlashTile - 1) / PtxKernels.FlashTile), (uint)heads, 1, 128, 1,
            P(q), P(keys), P(values), P(position), P(y), logSumExp is null ? 0UL : P(logSumExp),
            U(rowsPerHead), U(steps), U(capacity), U(dim), F(scale), U(variant.Window), F(variant.Softcap));
    }

    private const float Log2E = 1.4426950408889634f;

    // Strides of the tensor-core flash kernels for contiguous [heads, rowsPerHead, dim] queries / outputs and
    // [heads, capacity, dim] keys / values (kv = 1: every head is its own batch).
    // Packed sequences add each position's sequence start and end and the heads per packed row (0 without).
    private static ulong[] ContiguousLayout(int rowsPerHead, int steps, int capacity, int dim, Storage? starts = null, Storage? ends = null, int headsPerRow = 0) =>
        [U(1), U(rowsPerHead * dim), 0UL, U(steps * dim), U(dim), U(rowsPerHead * dim), 0UL, U(steps * dim), U(dim), U(capacity * dim), 0UL, U(dim),
            starts is null ? 0UL : P(starts), ends is null ? 0UL : P(ends), U(headsPerRow)];

    // Strides of the tensor-core flash kernels over q [heads, rowsPerHead, dim] and a cache's keys and values [heads,
    // capacity, row] (row: its elements of four bytes, dim floats or a bfloat16 cache's words), the output as q is, or with
    // `merged` key/value heads per sequence token-major (AttentionVariant.Merged: [sequences, steps, heads·dim]; head
    // hh = b · merged + h, row i = g · steps + t at (b · steps + t) · width + (h · group + g) · dim).
    private static ulong[] CacheLayout(int rowsPerHead, int steps, int capacity, int dim, int row, int merged)
    {
        if (merged <= 0)
        {
            return [U(1), U(rowsPerHead * dim), 0UL, U(steps * dim), U(dim), U(rowsPerHead * dim), 0UL, U(steps * dim), U(dim), U(capacity * row), 0UL, U(row),
                0UL, 0UL, 0UL];
        }

        int group = rowsPerHead / steps, width = merged * group * dim;
        return [U(merged), U(merged * rowsPerHead * dim), U(rowsPerHead * dim), U(steps * dim), U(dim), U(steps * width), U(group * dim), U(dim), U(width),
            U(merged * capacity * row), U(capacity * row), U(row), 0UL, 0UL, 0UL];
    }

    // Only the tensor-core flash passes write token-major here (MergesAttention): a call asking another path to is a caller's fault.
    private static void NotMerged(AttentionVariant variant, string operation)
    {
        if (variant.Merged > 0)
        {
            throw new InvalidOperationException($"{operation} was asked for token-major output (AttentionVariant.Merged) on a path that does not write it; ask Backend.MergesAttention first.");
        }
    }

    // The tensor-core flash passes write their output token-major (plain causal attention, bfloat16 products, head sizes
    // 64 and 128); the bfloat16 cache's needs its own kernel loaded.
    public override bool MergesAttention(Operation operation, int dim, AttentionVariant variant) =>
        variant.IsPlain && (operation == Ops.AttentionTiled || operation == Ops.AttentionBFloat16) && FlashTensorCore(dim) is not null
        && (operation != Ops.AttentionBFloat16 || dim <= PtxKernels.FlashMaxDim && TensorKernel(FlashHalvesName(dim)) is not null);

    // The tensor-core flash forward over a bfloat16 cache's rows, per head size.
    private static string FlashHalvesName(int dim) => dim switch
    {
        64 => "flash_tc_fwd_bf16_d64",
        128 => "flash_tc_fwd_bf16_d128",
        _ => $"flash_tc_fwd_bf16_d{dim}",
    };

    // Strides for [batch, steps, *] rows (see Backend.AttentionStrided).
    private static ulong[] RowLayout(int kvHeads, int group, int steps, int dim, int qRow, int kRow, int yRow) =>
        [U(kvHeads), U(steps * qRow), U(group * dim), U(dim), U(qRow), U(steps * yRow), U(group * dim), U(dim), U(yRow),
            U(steps * kRow), U(dim), U(kRow), 0UL, 0UL, 0UL];

    private Storage ZeroPosition => _zeroPosition ??= Allocate(1, zeroed: true);

    private Storage? _zeroPosition;

    bool IMatrixUnitBackend.AttentionStrided(Storage q, long qOffset, Storage k, long kOffset, Storage v, long vOffset, int qRow, int kRow,
        Storage y, Storage? logSumExp, int batch, int kvHeads, int group, int steps, int dim, float scale)
    {
        // The bfloat16 flash kernels; an fp8 run (no fp8 flash kernels) gets the caller's composed attention (its products
        // on the fp8 kernels or in float32, counted), never the bfloat16 kernels in fp8's place.
        if (MixedPrecision.Current == MatMulPrecision.Float8 || !PtxKernels.FlashTensorDim(dim) || !FlashKernelsLoaded(dim))
        {
            return false;
        }

        var tc = _tensorCoreAny;
        int heads = batch * kvHeads, rowsPerHead = group * steps;
        Launch(tc[FlashNames(dim).Forward], (uint)((rowsPerHead + PtxKernels.FlashTensorRows - 1) / PtxKernels.FlashTensorRows), (uint)heads, 1, 128, 1,
            [P(q) + (ulong)qOffset * 4, P(k) + (ulong)kOffset * 4, P(v) + (ulong)vOffset * 4, P(ZeroPosition), P(y), logSumExp is null ? 0UL : P(logSumExp),
            U(rowsPerHead), U(steps), U(steps), F(scale * Log2E), .. RowLayout(kvHeads, group, steps, dim, qRow, kRow, kvHeads * group * dim)]);
        return true;
    }

    bool IMatrixUnitBackend.AttentionStridedBackward(Storage q, long qOffset, Storage k, long kOffset, Storage v, long vOffset, int qRow, int kRow,
        Storage y, Storage logSumExp, Storage dOutput, Storage dq, long dqOffset, Storage dk, long dkOffset, Storage dv, long dvOffset,
        int batch, int kvHeads, int group, int steps, int dim, float scale)
    {
        if (MixedPrecision.Current == MatMulPrecision.Float8 || !PtxKernels.FlashTensorDim(dim) || !FlashKernelsLoaded(dim))   // as AttentionStrided
        {
            return false;
        }

        var tc = _tensorCoreAny;
        int heads = batch * kvHeads, rowsPerHead = group * steps, rows = heads * rowsPerHead, yRow = kvHeads * group * dim;
        var layout = RowLayout(kvHeads, group, steps, dim, qRow, kRow, yRow);
        var delta = Allocate(rows, zeroed: false);
        try
        {
            Launch1D(tc["flash_tc_delta"], rows, P(y), P(dOutput), P(delta), U(rowsPerHead), U(steps), U(dim), U(rows),
                U(kvHeads), layout[5], layout[6], layout[7], layout[8]);
            t_sharedBytes = (uint)PtxKernels.FlashTensorBackwardKvShared(dim);
            Launch(tc[FlashNames(dim).BackwardKv], (uint)((steps + PtxKernels.FlashTensorRows - 1) / PtxKernels.FlashTensorRows), (uint)heads, 1, 128, 1,
                [P(q) + (ulong)qOffset * 4, P(k) + (ulong)kOffset * 4, P(v) + (ulong)vOffset * 4, P(dOutput), P(logSumExp), P(delta),
                P(dk) + (ulong)dkOffset * 4, P(dv) + (ulong)dvOffset * 4,
                U(rowsPerHead), U(steps), U(steps), F(scale), F(scale * Log2E), U(group == 1 ? 1 : 0), .. layout]);
            t_sharedBytes = (uint)PtxKernels.FlashTensorBackwardQShared(dim);
            Launch(tc[FlashNames(dim).BackwardQ], (uint)((rowsPerHead + PtxKernels.FlashTensorRows - 1) / PtxKernels.FlashTensorRows), (uint)heads, 1, 128, 1,
                [P(q) + (ulong)qOffset * 4, P(k) + (ulong)kOffset * 4, P(v) + (ulong)vOffset * 4, P(dOutput), P(logSumExp), P(delta),
                P(dq) + (ulong)dqOffset * 4, U(rowsPerHead), U(steps), U(steps), F(scale), F(scale * Log2E), .. layout]);
        }
        finally
        {
            delta.Release();
        }

        return true;
    }

    // Chunks of rows in parallel; several chunks' sums are added in chunk order by fold_chunks_f32 (rule 86: no atomics).
    public override void SumColumnsKernel(Storage x, long offset, int ld, Storage y, int rows, int cols)
    {
        const int Chunk = 64;
        int chunks = (rows + Chunk - 1) / Chunk;
        var partials = chunks > 1 ? Allocate(chunks * cols, zeroed: false) : null;
        try
        {
            Launch(K("sum_cols_strided_f32"), (uint)((cols + _shapes.BlockSize - 1) / _shapes.BlockSize), (uint)chunks, 1, (uint)_shapes.BlockSize, 1,
                P(x) + (ulong)offset * 4, P(y), partials is null ? P(y) : P(partials), U(rows), U(cols), U(ld), U(Chunk));
            if (partials is not null)
            {
                Launch1D(K("fold_chunks_f32"), cols, P(partials), P(y), U(cols), U(chunks), U(cols));
            }
        }
        finally
        {
            partials?.Release();                                               // reused in stream order
        }
    }

    // The tensor-core flash kernels (bfloat16) when MixedPrecision asks for bfloat16 and the head size and GPU allow them.
    // A call that runs (`running`; null for a question) at a head size they serve, in a reduced precision without them
    // (fp8 and float16 have no flash kernels here; bfloat16 where they did not load), takes the float32 kernels, counted
    // (ReducedPrecision): never the bfloat16 kernels in another type's place.
    private Dictionary<string, IntPtr>? FlashTensorCore(int dim, Operation? running = null, long elements = 0)
    {
        if (MixedPrecision.Current == MatMulPrecision.Float32 || !PtxKernels.FlashTensorDim(dim))
        {
            return null;
        }

        if (BFloat16Products && FlashKernelsLoaded(dim))
        {
            return _tensorCoreAny;
        }

        if (running is not null)
        {
            ReducedPrecision.InFloat32(this, running, elements);
        }

        return null;
    }

    private bool FlashKernelsLoaded(int dim) =>
        TensorKernel(FlashNames(dim).Forward) is not null && TensorKernel(FlashNames(dim).BackwardQ) is not null
        && TensorKernel(FlashNames(dim).BackwardKv) is not null && TensorKernel("flash_tc_delta") is not null;

    // The flash kernels' names per head size (PtxKernels.FlashTensorDim: 64 and 128), built once: no name is formatted per launch.
    private static readonly (string Forward, string BackwardQ, string BackwardKv) Flash64 = ("flash_tc_fwd_d64", "flash_tc_bwd_q_d64", "flash_tc_bwd_kv_d64");
    private static readonly (string Forward, string BackwardQ, string BackwardKv) Flash128 = ("flash_tc_fwd_d128", "flash_tc_bwd_q_d128", "flash_tc_bwd_kv_d128");

    private static (string Forward, string BackwardQ, string BackwardKv) FlashNames(int dim) => dim switch
    {
        64 => Flash64,
        128 => Flash128,
        _ => ($"flash_tc_fwd_d{dim}", $"flash_tc_bwd_q_d{dim}", $"flash_tc_bwd_kv_d{dim}"),
    };

    public override void AttentionTiledBackwardKernel(Storage q, Storage keys, Storage values, Storage output, Storage logSumExp, Storage dOutput,
        Storage dq, Storage dkeys, Storage dvalues, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale, AttentionVariant variant = default)
    {
        int rows = heads * rowsPerHead;
        var delta = Allocate(rows, zeroed: false);
        try
        {
            Launch1D(K("attn_bwd_d_f32"), rows, P(output), P(dOutput), P(delta), U(dim), U(rows));
            if (variant.IsPlain && FlashTensorCore(dim, Ops.AttentionTiledBackward, (long)heads * (rowsPerHead + 2L * steps) * dim) is { } tc)
            {
                var layout = ContiguousLayout(rowsPerHead, steps, capacity, dim);
                t_sharedBytes = (uint)PtxKernels.FlashTensorBackwardKvShared(dim);
                Launch(tc[FlashNames(dim).BackwardKv], (uint)((capacity + PtxKernels.FlashTensorRows - 1) / PtxKernels.FlashTensorRows), (uint)heads, 1, 128, 1,
                    [P(q), P(keys), P(values), P(dOutput), P(logSumExp), P(delta), P(dkeys), P(dvalues),
                    U(rowsPerHead), U(steps), U(capacity), F(scale), F(scale * Log2E), U(rowsPerHead == steps ? 1 : 0), .. layout]);
                t_sharedBytes = (uint)PtxKernels.FlashTensorBackwardQShared(dim);
                Launch(tc[FlashNames(dim).BackwardQ], (uint)((rowsPerHead + PtxKernels.FlashTensorRows - 1) / PtxKernels.FlashTensorRows), (uint)heads, 1, 128, 1,
                    [P(q), P(keys), P(values), P(dOutput), P(logSumExp), P(delta), P(dq),
                    U(rowsPerHead), U(steps), U(capacity), F(scale), F(scale * Log2E), .. layout]);
                return;
            }

            ReadOnlySpan<ulong> args = [P(q), P(keys), P(values), P(dOutput), P(logSumExp), P(delta), P(dq), P(dkeys), P(dvalues),
                U(rowsPerHead), U(steps), U(capacity), U(dim), F(scale), U(variant.Window), F(variant.Softcap)];
            Launch(K("attn_bwd_kv_f32"), (uint)((capacity + 15) / 16), (uint)heads, 1, 128, 1, args);
            Launch(K("attn_bwd_q_f32"), (uint)((rowsPerHead + 31) / 32), (uint)heads, 1, 128, 1, args);
        }
        finally
        {
            delta.Release();
        }
    }

    // Packed sequences run on the tensor-core flash kernels, which compute plain causal attention only: a window or a cap
    // is refused (fine-tuning then pads).
    bool ISegmentedAttentionBackend.SupportsSegmented(int dim, AttentionVariant variant) => variant.IsPlain && FlashTensorCore(dim) is not null;

    // Rows of different lengths: float32 caches on the flash kernels where they serve, else on the decoding kernel with
    // per-row starts (attention_rows_f32); bfloat16 caches on attention_rows_bf16 (and attention_flash_rows_bf16 for
    // prompt-sized steps). Both decoding kernels take every head size up to DecodeMaxDim, windows and caps, and the
    // tensor-core flash head sizes lie within it: so the head sizes both cache formats run on the device.
    bool IRowAttentionBackend.SupportsRows(int dim, AttentionVariant variant) => dim <= PtxKernels.DecodeMaxDim;

    bool IRowAttentionBackend.AttentionRows(Storage q, Storage keys, Storage values, Storage position, Storage y, Storage starts, int heads, int headsPerRow,
        int rowsPerHead, int steps, int capacity, int dim, float scale, AttentionVariant variant)
    {
        if (!variant.IsPlain || FlashTensorCore(dim, Ops.AttentionRows, (long)heads * (rowsPerHead + 2L * steps) * dim) is not { } tc)
        {
            if (dim > PtxKernels.DecodeMaxDim)
            {
                return false;
            }

            // The decoding kernel, each row also from its sequence's start (its splits measured as a kernel of its own).
            DecodeSplit(WindowedTuning(6, variant, capacity),
                new DecodeAttention(3, q, keys, values, null, null, y, rowsPerHead, steps, capacity, dim, scale, 0, variant, heads * rowsPerHead, starts, headsPerRow),
                position);
            return true;
        }

        Launch(tc[FlashNames(dim).Forward], (uint)((rowsPerHead + PtxKernels.FlashTensorRows - 1) / PtxKernels.FlashTensorRows), (uint)heads, 1, 128, 1,
            [P(q), P(keys), P(values), P(position), P(y), 0UL, U(rowsPerHead), U(steps), U(capacity), F(scale * Log2E),
            .. ContiguousLayout(rowsPerHead, steps, capacity, dim, starts, starts, headsPerRow)]);
        return true;
    }

    bool ISegmentedAttentionBackend.AttentionSegmented(Storage q, Storage keys, Storage values, Storage y, Storage? logSumExp, Storage starts, Storage ends,
        int heads, int headsPerRow, int rowsPerHead, int steps, int dim, float scale, AttentionVariant variant)
    {
        if (!variant.IsPlain || FlashTensorCore(dim) is not { } tc)
        {
            return false;
        }

        Launch(tc[FlashNames(dim).Forward], (uint)((rowsPerHead + PtxKernels.FlashTensorRows - 1) / PtxKernels.FlashTensorRows), (uint)heads, 1, 128, 1,
            [P(q), P(keys), P(values), P(ZeroPosition), P(y), logSumExp is null ? 0UL : P(logSumExp),
            U(rowsPerHead), U(steps), U(steps), F(scale * Log2E), .. ContiguousLayout(rowsPerHead, steps, steps, dim, starts, ends, headsPerRow)]);
        return true;
    }

    bool ISegmentedAttentionBackend.AttentionSegmentedBackward(Storage q, Storage keys, Storage values, Storage output, Storage logSumExp, Storage dOutput,
        Storage dq, Storage dkeys, Storage dvalues, Storage starts, Storage ends, int heads, int headsPerRow, int rowsPerHead, int steps, int dim, float scale,
        AttentionVariant variant)
    {
        if (!variant.IsPlain || FlashTensorCore(dim) is not { } tc)
        {
            return false;
        }

        int rows = heads * rowsPerHead;
        var delta = Allocate(rows, zeroed: false);
        try
        {
            Launch1D(K("attn_bwd_d_f32"), rows, P(output), P(dOutput), P(delta), U(dim), U(rows));
            var layout = ContiguousLayout(rowsPerHead, steps, steps, dim, starts, ends, headsPerRow);
            t_sharedBytes = (uint)PtxKernels.FlashTensorBackwardKvShared(dim);
            Launch(tc[FlashNames(dim).BackwardKv], (uint)((steps + PtxKernels.FlashTensorRows - 1) / PtxKernels.FlashTensorRows), (uint)heads, 1, 128, 1,
                [P(q), P(keys), P(values), P(dOutput), P(logSumExp), P(delta), P(dkeys), P(dvalues),
                U(rowsPerHead), U(steps), U(steps), F(scale), F(scale * Log2E), 0UL, .. layout]);
            t_sharedBytes = (uint)PtxKernels.FlashTensorBackwardQShared(dim);
            Launch(tc[FlashNames(dim).BackwardQ], (uint)((rowsPerHead + PtxKernels.FlashTensorRows - 1) / PtxKernels.FlashTensorRows), (uint)heads, 1, 128, 1,
                [P(q), P(keys), P(values), P(dOutput), P(logSumExp), P(delta), P(dq),
                U(rowsPerHead), U(steps), U(steps), F(scale), F(scale * Log2E), .. layout]);
        }
        finally
        {
            delta.Release();
        }

        return true;
    }

    public override void SoftmaxCrossEntropyRowsKernel(Storage logits, Storage targets, Storage weights, Storage losses, int rows, int vocabulary, float scale) =>
        LaunchRows(K("softmax_ce_rows_f32"), rows, PtxKernels.SoftmaxCrossEntropyThreads, P(logits), P(targets), P(weights), P(losses), U(vocabulary), F(scale), U(rows));

    /// <summary>Tests and benchmarks: the path of rows over a bfloat16 cache (0 the decoding kernel, 1 the tiled one) instead of the measured one.</summary>
    internal static int? RowsBFloat16Path { get; set; }

    // Steps from which the bfloat16 cache's own attention takes its tiled kernel (KeyValueLayouts: tiled when steps >= 8);
    // rows of different lengths start from the same choice until the two paths are measured.
    private const int TiledSteps = 8;

    // The path of AttentionRowsBFloat16Kernel (0 decoding, 1 tiled): forced, the only one, measured or the default.
    private int RowsBFloat16Choice(int tuning, in DecodeAttention a, Storage position)
    {
        bool tiled = a.Dim <= PtxKernels.FlashMaxDim;
        if (RowsBFloat16Path is int forced)
        {
            return tiled ? Math.Clamp(forced, 0, 1) : 0;
        }

        if (!tiled || a.Steps == 1)
        {
            return 0;
        }

        var key = new TuneKey(TuneOp.RowsPath, tuning, TuneSizes.Class(a.Rows / a.RowsPerHead), TuneSizes.Class(a.RowsPerHead), TuneSizes.Class(a.Capacity), a.Dim);
        int formula = a.Steps >= TiledSteps ? 1 : 0;
        ReadOnlySpan<int> candidates = [0, 1];
        if (TryTuned(key, candidates, formula, out int path))
        {
            return path;                                                       // known: no closure, no allocation
        }

        var operands = a;
        return Tune(key, candidates, formula, c => RowsBFloat16(c, tuning, operands, position));   // each writes y, which the chosen one writes again
    }

    private void RowsBFloat16(int path, int tuning, in DecodeAttention a, Storage position)
    {
        if (path == 0)
        {
            DecodeSplit(tuning, a, position);
            return;
        }

        Launch(K("attention_flash_rows_bf16"), (uint)((a.RowsPerHead + PtxKernels.FlashTile - 1) / PtxKernels.FlashTile), (uint)(a.Rows / a.RowsPerHead), 1, 128, 1,
            P(a.Q), P(a.Keys), P(a.Values), P(position), P(a.Y), 0UL, U(a.RowsPerHead), U(a.Steps), U(a.Capacity), U(a.Dim), F(a.Scale), U(a.Words),
            U(a.Variant.Window), F(a.Variant.Softcap), P(a.Starts!), U(a.HeadsPerRow));
    }

    public override void AddRmsNormAffineKernel(Storage a, Storage b, Storage sum, Storage gain, Storage y, int rows, int cols, float eps, float offset) =>
        LaunchRows(K("add_rms_norm_affine_f32"), rows, P(a), P(b), P(sum), P(gain), P(y), U(cols), F(eps), F(offset), U(rows));

    public override void RmsNormRopeKernel(Storage x, Storage gain, Storage cos, Storage sin, Storage positions, Storage y, int rows, int cols,
        float eps, float offset, int heads, int steps, int half, bool interleaved) =>
        LaunchRows(K("rms_norm_rope_f32"), rows, P(x), P(gain), P(cos), P(sin), P(positions), P(y),
            U(cols), F(eps), F(offset), U(heads), U(steps), U(half), U(interleaved ? 1 : 0), U(rows));

    public override void RmsNormRopePairKernel(Storage x, Storage gain, Storage y, int rows1, float eps, float offset, int heads,
        Storage x2, Storage gain2, Storage y2, int rows2, float eps2, float offset2, int heads2,
        Storage cos, Storage sin, Storage positions, int cols, int steps, int half, bool interleaved) =>
        LaunchRows(K("rms_norm_rope2_f32"), rows1 + rows2, P(x), P(gain), P(cos), P(sin), P(positions), P(y), P(x2), P(gain2), P(y2),
            U(cols), F(eps), F(offset), U(heads), U(steps), U(half), U(interleaved ? 1 : 0), U(rows1), F(eps2), F(offset2), U(heads2),
            U(rows1 + rows2));

    public override bool NormRopeHeadsKernel(Storage q, Storage k, Storage v, int batch, int steps, int heads, int kvHeads, int cols,
        Storage? gainQ, float epsQ, float offsetQ, Storage? gainK, float epsK, float offsetK, Storage? cos, Storage? sin, Storage? positions,
        int half, bool interleaved, Storage yq, Storage yk, Storage yv, Storage? position, int capacity, int stride, bool bfloat16)
    {
        int rows1 = batch * steps * heads, rows2 = batch * steps * kvHeads;
        int flags = (gainQ is not null ? 1 : 0) | (bfloat16 ? 2 : 0) | (position is not null ? 4 : 0);
        static ulong Optional(Storage? s) => s is null ? 0UL : P(s);
        ReadOnlySpan<ulong> args = [P(q), Optional(gainQ), P(k), Optional(gainK), P(v), Optional(cos), Optional(sin),
            Optional(positions), Optional(position), P(yq), P(yk), P(yv), U(cols), F(epsQ), F(offsetQ), U(heads), F(epsK), F(offsetK), U(kvHeads),
            U(steps), U(cos is null ? 0 : half), U(interleaved ? 1 : 0), U(rows1), U(rows2), U(flags), U(capacity), U(stride), U(rows1 + 2 * rows2)];

        // A row is one head: its block's threads measured once per head size and row count class (about one thread a pair of
        // values before), since 256 threads over 64 or 128 values leave most idle and pay the block's barriers for nothing.
        // Each run writes the outputs (and cache slots) the final run writes again.
        int rows = rows1 + 2 * rows2, formula = RowThreadsFor(cols);
        var key = new TuneKey(TuneOp.RowThreads, 0, cols, TuneSizes.Class(rows), interleaved ? 1 : 0);
        if (!TryTuned(key, RowThreadCounts, formula, out int threads))
        {
            threads = MeasureRowThreads(key, formula, K("norm_rope_heads_f32"), rows, args.ToArray());
        }

        LaunchRows(K("norm_rope_heads_f32"), rows, threads, args);
        return true;
    }

    // Block sizes a short-row kernel is measured with (each a whole number of warps the reductions take).
    private static readonly int[] RowThreadCounts = [32, 64, 128, 256];

    // Before measuring: the power of two near half the row's length, from one warp to RowThreads.
    private static int RowThreadsFor(int cols) => Math.Clamp((int)BitOperations.RoundUpToPowerOf2((uint)Math.Max(1, cols / 2)), 32, PtxKernels.RowThreads);

    [MethodImpl(MethodImplOptions.NoInlining)]
    private int MeasureRowThreads(TuneKey key, int formula, IntPtr function, int rows, ulong[] args) =>
        Tune(key, RowThreadCounts, formula, threads => LaunchRows(function, rows, threads, args));
}
