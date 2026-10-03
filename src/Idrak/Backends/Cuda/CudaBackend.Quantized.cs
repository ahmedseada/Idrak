// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;
using static Idrak.Backends.Cuda.CudaDriver;

namespace Idrak.Backends.Cuda;

// Int8 weight-only quantization kernels (see PtxKernels.Quantized.cs).
internal sealed unsafe partial class CudaBackend
{
    public override void Int8MatMul(Storage x, Storage q, Storage scales, Storage y, int m, int n, int k)
    {
        int words = (n + 3) / 4;
        if (m > PtxKernels.GemvRows || k == 0)
        {
            Launch1D(K("int8_matmul_f32"), m * words, P(x), P(q), P(scales), P(y), U(k), U(words), U(n), U(m * words));
            return;
        }

        PackedFewRows((int)PackedFormat.Int8 * 4 + GemvPlain, x, q, scales, y, m, n, k, words);
    }

    public override void BFloat16MatMul(Storage x, Storage packed, Storage y, int m, int n, int k)
    {
        int words = (n + 1) / 2;
        if (m > PtxKernels.GemvRows && PackedMatMulLarge(PackedFormat.BFloat16, x, packed, null, y, m, n, k))
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

    public override void Int4MatMul(Storage x, Storage q, Storage scales, Storage y, int m, int n, int k)
    {
        int words = (n + 7) / 8;
        if (m > PtxKernels.GemvRows && PackedMatMulLarge(PackedFormat.Int4, x, q, scales, y, m, n, k))
        {
            return;
        }

        if (m > PtxKernels.GemvRows || k == 0)
        {
            var w = Allocate(k * n, zeroed: false);
            try
            {
                Int4Dequantize(q, scales, w, k, n);
                MatMul(x, w, y, m, n, k, false, false, 0f);
            }
            finally
            {
                w.Release();
            }

            return;
        }

        PackedFewRows((int)PackedFormat.Int4 * 4 + GemvPlain, x, q, scales, y, m, n, k, words, align: 64);
    }

    public override void Int4Dequantize(Storage q, Storage scales, Storage w, int k, int n)
    {
        int words = (n + 7) / 8;
        Launch1D(K("int4_dequant_f32"), k * words, P(q), P(scales), P(w), U(words), U(n), U(k * words));
    }

    public override void BFloat16Dequantize(Storage packed, Storage w, int k, int n)
    {
        int words = (n + 1) / 2;
        Launch1D(K("bf16_dequant_f32"), k * words, P(packed), P(w), U(words), U(n), U(k * words));
    }

    /// <summary>Benchmarks only: the k splits of prompt-sized packed products on tensor cores instead of the heuristic's.</summary>
    internal static int? PackedSplits { get; set; }

    /// <summary>Benchmarks only: the number of blocks decoding attention splits the cache over instead of the heuristic's.</summary>
    internal static int? DecodeSplits { get; set; }

    /// <summary>Benchmarks only: the number of k splits of the few-row packed products instead of the heuristic's.</summary>
    internal static int? GemvSplits { get; set; }

    // Kernel names by format (0 int8, 1 int4, 2 bfloat16) and variant, built once: the per-token path never formats a name.
    internal const int GemvPlain = 0, GemvSilu = 1, GemvGelu = 2, GemvAddNorm = 3;
    private static readonly string[] GemvKernels =
    [
        "int8_gemv_f32", "int8_gemv_silu_f32", "int8_gemv_gelu_f32", "int8_gemv_addnorm_f32",
        "int4_gemv_f32", "int4_gemv_silu_f32", "int4_gemv_gelu_f32", "int4_gemv_addnorm_f32",
        "bf16_gemv_f32", "bf16_gemv_silu_f32", "bf16_gemv_gelu_f32", "bf16_gemv_addnorm_f32",
    ];

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
    // GemvAddNorm) and the shape, so the fused kernels never take the plain kernel's choice.
    internal static TuneKey GemvSplitsKey(int variant, int m, int n, int k) => new(TuneOp.GemvSplits, variant, m, n, k);

    // Split counts the few-row products are measured with: 1, 2, 3, 4, 6, 8, 12, ... up to GemvMaxSplits (the kernels
    // take any chunk of k: a step of 64 rows, then single rows; --bench-gemv found 6 best on an RTX 3060 Laptop for the
    // products into 1024 columns, which the powers of two alone missed by 25%), without counts that give the same
    // chunks once rounded to `align` (int4: a 64-row group); of two such counts the power of two stays, so the
    // formula's choice (GemvSplitCount) is always one of them.
    internal static int[] GemvSplitCandidates(int k, int align)
    {
        var values = new List<int>();
        var byCount = new Dictionary<int, int>();                      // splits that run → index in values
        foreach (int wanted in SplitCounts(GemvMaxSplits(k)))
        {
            int chunk = ((k + wanted - 1) / wanted + align - 1) / align * align;
            int count = (k + chunk - 1) / chunk;
            if (!byCount.TryGetValue(count, out int at))
            {
                byCount[count] = values.Count;
                values.Add(wanted);
            }
            else if (int.IsPow2(wanted))
            {
                values[at] = wanted;
            }
        }

        return [.. values];
    }

    // Few rows (decoding) through packed weights: read each weight word once, with enough blocks to keep every
    // multiprocessor busy; narrow matrices split k, and the last block of each column range adds the splits in order.
    // `align`: split boundaries fall on multiples of it (int4 splits start on a 64-row block).
    // `up`: the gated kernels' second input; `tail`: further arguments (the addnorm kernels').
    // `variant`: format * 4 + GemvPlain / GemvSilu / GemvGelu / GemvAddNorm (an index into GemvKernels).
    private void PackedFewRows(int variant, Storage x, Storage q, Storage scales, Storage y, int m, int n, int k, int words, int align = 1,
        Storage? up = null, ulong[]? tail = null)
    {
        string kernel = GemvKernels[variant];
        int columnBlocks = (words + 31) / 32;
        var counters = SplitCounters(columnBlocks + 1);
        void Run(string name, ulong output, ulong[] extra, int wanted)
        {
            int chunk = ((k + wanted - 1) / wanted + align - 1) / align * align;
            int count = (k + chunk - 1) / chunk;
            var part = count > 1 ? Allocate(count * m * n, zeroed: false) : null;
            try
            {
                ulong[] args = [P(x), P(q), P(scales), output, part is null ? output : P(part), U(m), U(n), U(k), U(words), U(chunk), U(count),
                    P(counters), .. up is null ? [] : new[] { P(up) }, .. extra];
                Launch(K(name), (uint)columnBlocks, (uint)count, 1, PtxKernels.Int8GemvThreads, 1, args);
            }
            finally
            {
                part?.Release();
            }
        }

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
            // without that last step: --bench-gemv showed 8 splits chosen where 16 was 15% faster on an RTX 5070 Ti.)
            int formula = GemvSplitCount(columnBlocks, k);
            int[] candidates = GemvSplitCandidates(k, align);
            var key = GemvSplitsKey(variant, m, n, k);
            splits = formula;
            if (tail is null)
            {
                splits = Tune(key, candidates, formula, c => Run(kernel, P(y), [], c), cold: true);
            }
            else if (!TunedKnown(key))
            {
                long size = (long)m * n * sizeof(float);
                WithScratch(3L * m * n, scratch => splits = Tune(key, candidates, formula,
                    c => Run(kernel, scratch, [tail[0], scratch + (ulong)size, tail[2], scratch + 2 * (ulong)size, .. tail[4..]], c), cold: true));
            }
            else
            {
                splits = Tune(key, [], formula, _ => { });
            }
        }

        Run(kernel, P(y), tail ?? [], splits);
    }

    /// <summary>
    /// Tests and benchmarks: with a value, 4-8 int8 rows go to the packed product from this many weights (k·n) on, without
    /// measuring; null (the default): measured per shape on this card.
    /// </summary>
    internal static long? PackedPreferredWeights { get; set; }

    // Few int8 rows through the GEMV (its time grows with the rows) or the packed tensor-core product (flat in the rows,
    // but a mostly empty row tile): where they cross depends on the card (on compute 12.0 cards the product won from 4 rows
    // through large layers, on 8.6 it lost), so both run once per shape and the faster is kept. Until then, and when
    // nothing can be measured, the GEMV.
    public override bool PrefersPackedMatMul(PackedFormat format, Storage x, Storage packed, Storage? scales, Storage y, int m, int n, int k)
    {
        if (format != PackedFormat.Int8 || m < 2 || n < 64 || k < 32 || !MixedPrecision.UsesTensorCores)
        {
            return false;
        }

        if (PackedPreferredWeights is long threshold)
        {
            return m >= 4 && (long)k * n >= threshold;
        }

        void Run(int choice)
        {
            if (choice == 0 || !PackedMatMulLarge(0, x, packed, scales, y, m, n, k))
            {
                Int8MatMul(x, packed, scales ?? packed, y, m, n, k);
            }
        }

        return Tune(new TuneKey(TuneOp.Int8FewRows, 0, m, n, k), [0, 1], 0, Run, cold: true) == 1;
    }

    /// <summary>Benchmarks only: the row tile (64 or 128) of prompt-sized packed products instead of the heuristic's.</summary>
    internal static int? PromptTileRowsOverride { get; set; }

    // Row tile of a prompt-sized packed product: 64 when the last 128-row tile would be at most half full (180 rows: 3
    // tiles of 64, 192 rows computed, instead of 2 of 128, 256), else 128.
    private static int PromptTileRows(int m) =>
        PromptTileRowsOverride ?? (m < 1024 && m % PtxKernels.TensorTile is > 0 and <= 64 ? 64 : PtxKernels.TensorTile);

    // k splits of a prompt-sized packed product before measuring (see PromptSplitsTuned): up to four blocks per SM, chunks
    // of 256 k or more; none when the tiles already fill about one wave of the SMs (splitting then only adds the zeroing
    // and atomic additions).
    private int PromptSplits(int tiles, int k)
    {
        int sms = Math.Max(1, _multiprocessors);
        return tiles * 100 >= sms * 85 && tiles <= sms ? 1 : Math.Clamp(Math.Min(k / 256, 4 * sms / tiles), 1, 8);
    }

    // The k splits of a prompt-sized packed product, measured once per shape on this card: `run(splits)` zeroes the outputs
    // when it splits and launches, writing outputs the final run writes again. With four or more waves of tiles there is
    // nothing to gain from splitting, so nothing is measured.
    private int PromptSplitsTuned(TuneKey key, int tiles, int k, Action<int> run)
    {
        int formula = PromptSplits(tiles, k);
        return tiles >= 4 * Math.Max(1, _multiprocessors) ? formula : Tune(key, SplitCounts(Math.Min(16, k / 128)), formula, run);
    }

    public override bool PackedMatMulLarge(PackedFormat format, Storage x, Storage packed, Storage? scales, Storage y, int m, int n, int k)
    {
        // Any row count the caller sends (above the GEMV kernels' limit, or fewer rows when PrefersPackedMatMul): short
        // prompts fill part of one row tile, which the kernels bound-check, instead of expanding the whole weight to float32
        // first (that writes and reads four bytes per weight on every card, a pass the packed product never makes).
        if (m < 1 || n < 64 || k < 8)
        {
            return false;
        }

        // Tensor cores (MixedPrecision): the weights unpacked into the bfloat16 tiles as they are loaded.
        int tileRows = PromptTileRows(m);
        int packedVariant = (int)format * 2 + (tileRows == 64 ? 1 : 0);
        string packedKernel = PackedTensorKernels[packedVariant];
        if (MixedPrecision.UsesTensorCores && k >= 32 && TensorKernel(packedKernel) is { } tensor)
        {
            int perWord = format.ValuesPerWord();
            if (_profile is not null)
            {
                _profileLabel = $"gemm_tc_nn_{format.KernelName() + "w"}{(tileRows == 64 ? "_m64" : "")} {m}x{n}x{k}";
                _profileFlops = 2.0 * m * n * k;
            }

            // Split k when the output tiles leave SMs idle (prompt-sized m): up to two blocks per SM, chunks of 256 k or
            // more, partial sums added into the zeroed output.
            int rowTiles = (m + tileRows - 1) / tileRows, columnTiles = (n + PtxKernels.TensorTile - 1) / PtxKernels.TensorTile;
            void Run(int splits)
            {
                if (splits > 1)
                {
                    Check(cuMemsetD32Async(P(y), 0, (nuint)((long)m * n), _stream), nameof(cuMemsetD32Async));
                }

                Launch(tensor, (uint)columnTiles, (uint)rowTiles,
                    (uint)splits, PtxKernels.TensorThreads, 1, P(x), P(packed), P(y), U(m), U(n), U(k), F(0f), 0UL, 0UL, 0UL, 0UL,
                    U(k), U((n + perWord - 1) / perWord), U(n), scales is null ? 0UL : P(scales));
            }

            Run(PackedSplits is int forced ? Math.Clamp(forced, 1, Math.Max(1, k / 32))
                : PromptSplitsTuned(new TuneKey(TuneOp.PackedSplits, packedVariant, m, n, k), rowTiles * columnTiles, k, Run));
            return true;
        }

        // Without tensor cores: 128 × 128 tiles when that still gives every SM a block, else 64 × 64, measured per shape.
        string formatName = format.KernelName();
        long tiles128 = (long)((m + 127) / 128) * ((n + 127) / 128);
        void RunTiles(int tile) =>
            Launch(K(PackedGemmKernels[(int)format * 2 + (tile == 128 ? 1 : 0)]), (uint)((n + tile - 1) / tile), (uint)((m + tile - 1) / tile), 1, PtxKernels.GemmThreads, 1,
                P(x), P(packed), P(y), U(m), U(n), U(k), U(0), U(0), F(0f), 0UL, 0UL, 0UL, P(scales ?? packed));
        RunTiles(Tune(new TuneKey(TuneOp.PackedTile, (int)format, m, n, k), [64, 128], tiles128 >= Math.Max(1, _multiprocessors) ? 128 : 64, RunTiles));
        return true;
    }

    // Prompt-sized products of one input through 2-3 packed layers (queries/keys/values, gate/up) in one tensor-core
    // launch: the column tiles of every product side by side, so a few rows still fill the GPU (180 rows are 2 row
    // tiles: q/k/v alone gave 32 tiles, 16 and 16 in separate launches). Widths must be multiples of the 128-column tile,
    // and the layers must have no bias (the caller adds none).
    private bool PackedManyLarge(PackedFormat format, Storage x, int m, int k,
        ReadOnlySpan<(Storage Packed, Storage? Scales, Storage? Bias, Storage Output, int Columns)> products)
    {
        if (products.Length is < 2 or > 3 || m < 64 || k < 32 || !MixedPrecision.UsesTensorCores)
        {
            return false;
        }

        int columnTiles = 0;
        foreach (var product in products)
        {
            if (product.Bias is not null || product.Columns % PtxKernels.TensorTile != 0 || product.Columns == 0)
            {
                return false;
            }

            columnTiles += product.Columns / PtxKernels.TensorTile;
        }

        int tileRows = PromptTileRows(m);
        string formatName = format.KernelName() + "w";
        int multiVariant = (int)format * 2 + (tileRows == 64 ? 1 : 0);
        if (TensorKernel(PackedMultiKernels[multiVariant]) is not { } tensor)
        {
            return false;
        }

        int perWord = format.ValuesPerWord();
        int rowTiles = (m + tileRows - 1) / tileRows;
        if (_profile is not null)
        {
            _profileLabel = $"gemm_tc_nn_{formatName}_multi{(tileRows == 64 ? "_m64" : "")} {m}x{string.Join('+', products.ToArray().Select(p => p.Columns))}x{k}";
            _profileFlops = 0;
            foreach (var product in products)
            {
                _profileFlops += 2.0 * m * product.Columns * k;
            }
        }

        static ulong Scales(Storage? scales) => scales is null ? 0UL : P(scales);
        var list = products.ToArray();                                     // prompt-sized: one small copy per call
        void Run(int splits)
        {
            if (splits > 1)
            {
                foreach (var product in list)
                {
                    Check(cuMemsetD32Async(P(product.Output), 0, (nuint)((long)m * product.Columns), _stream), nameof(cuMemsetD32Async));
                }
            }

            var p1 = list[1];
            var p2 = list.Length == 3 ? list[2] : list[1];
            int n2 = list.Length == 3 ? list[2].Columns : 0;
            Launch(tensor, (uint)columnTiles, (uint)rowTiles, (uint)splits, PtxKernels.TensorThreads, 1,
                P(x), P(list[0].Packed), P(list[0].Output), U(m), U(list[0].Columns), U(k), F(0f), 0UL, 0UL, 0UL, 0UL,
                U(k), U(list[0].Columns / perWord), U(list[0].Columns), Scales(list[0].Scales),
                P(p1.Packed), P(p1.Output), Scales(p1.Scales), U(p1.Columns),
                P(p2.Packed), P(p2.Output), Scales(p2.Scales), U(n2));
        }

        var key = new TuneKey(TuneOp.PackedMultiSplits, multiVariant, m, k, list[0].Columns, list[1].Columns, list.Length == 3 ? list[2].Columns : 0);
        Run(PackedSplits is int forced ? Math.Clamp(forced, 1, Math.Max(1, k / 32)) : PromptSplitsTuned(key, rowTiles * columnTiles, k, Run));
        return true;
    }

    public override bool PackedMatMulLowRank(PackedFormat format, Storage x, int m, int k,
        ReadOnlySpan<(Storage Packed, Storage? Scales, Storage Output, int Columns, Storage U, Storage V)> products, int rank)
    {
        if (products.Length is < 1 or > 3 || format is not (PackedFormat.Int4 or PackedFormat.BFloat16) || m < 64 || k < 32 || rank is < 1 or > 32 || !MixedPrecision.UsesTensorCores
            || MixedPrecision.Current == MatMulPrecision.Float8)
        {
            return false;
        }

        bool multi = products.Length > 1;
        int columnTiles = 0;
        foreach (var product in products)
        {
            if (product.Columns == 0 || multi && product.Columns % PtxKernels.TensorTile != 0)
            {
                return false;
            }

            columnTiles += (product.Columns + PtxKernels.TensorTile - 1) / PtxKernels.TensorTile;
        }

        string formatName = format.KernelName() + "w";
        int lowRankVariant = (format == PackedFormat.Int4 ? 0 : 2) + (multi ? 1 : 0);
        if (TensorKernel(PackedLowRankKernels[lowRankVariant]) is not { } tensor)
        {
            return false;
        }

        int perWord = format.ValuesPerWord();
        int rowTiles = (m + PtxKernels.TensorTile - 1) / PtxKernels.TensorTile;
        if (_profile is not null)
        {
            _profileLabel = $"gemm_tc_nn_{formatName}{(multi ? "_multi" : "")}_lr {m}x{string.Join('+', products.ToArray().Select(p => p.Columns))}x{k}+{rank}";
            _profileFlops = 0;
            foreach (var product in products)
            {
                _profileFlops += 2.0 * m * product.Columns * (k + rank);
            }
        }

        static ulong Scales(Storage? scales) => scales is null ? 0UL : P(scales);
        var list = products.ToArray();                                     // prompt-sized: one small copy per call
        void Run(int splits)
        {
            if (splits > 1)
            {
                foreach (var product in list)
                {
                    Check(cuMemsetD32Async(P(product.Output), 0, (nuint)((long)m * product.Columns), _stream), nameof(cuMemsetD32Async));
                }
            }

            var p0 = list[0];
            if (!multi)
            {
                Launch(tensor, (uint)columnTiles, (uint)rowTiles, (uint)splits, PtxKernels.TensorThreads, 1,
                    P(x), P(p0.Packed), P(p0.Output), U(m), U(p0.Columns), U(k), F(0f), 0UL, 0UL, 0UL, 0UL,
                    U(k), U((p0.Columns + perWord - 1) / perWord), U(p0.Columns), Scales(p0.Scales), P(p0.U), P(p0.V), U(rank));
                return;
            }

            var p1 = list[1];
            var p2 = list.Length == 3 ? list[2] : list[1];
            int n2 = list.Length == 3 ? list[2].Columns : 0;
            Launch(tensor, (uint)columnTiles, (uint)rowTiles, (uint)splits, PtxKernels.TensorThreads, 1,
                P(x), P(p0.Packed), P(p0.Output), U(m), U(p0.Columns), U(k), F(0f), 0UL, 0UL, 0UL, 0UL,
                U(k), U(p0.Columns / perWord), U(p0.Columns), Scales(p0.Scales),
                P(p1.Packed), P(p1.Output), Scales(p1.Scales), U(p1.Columns),
                P(p2.Packed), P(p2.Output), Scales(p2.Scales), U(n2),
                P(p0.U), P(p0.V), U(rank), P(p1.U), P(p1.V), P(p2.U), P(p2.V));
        }

        var key = new TuneKey(TuneOp.PackedLowRankSplits, lowRankVariant, m, k, list[0].Columns,
            list.Length > 1 ? list[1].Columns : 0, list.Length > 2 ? list[2].Columns : 0, rank);
        Run(PackedSplits is int forced ? Math.Clamp(forced, 1, Math.Max(1, k / 32)) : PromptSplitsTuned(key, rowTiles * columnTiles, k, Run));
        return true;
    }

    public override bool PackedMatMulGated(PackedFormat format, int activation, Storage gate, Storage up, Storage packed, Storage? scales, Storage y,
        int m, int n, int k)
    {
        if (m > PtxKernels.GemvRows || k == 0 || activation is not (0 or 1))
        {
            return false;
        }

        // The activation computed per input value inside the product (fused), or by its own pass first (separate): which
        // is faster depends on the card and the format (eight int4 columns per word repay the extra work sooner than four
        // int8 or two bfloat16 ones), so it is measured per shape; both write all of y.
        int cpw = format.ValuesPerWord();
        int words = (n + cpw - 1) / cpw, align = format.SplitAlignment();
        int fused = (int)format * 4 + (activation == 0 ? GemvSilu : GemvGelu);
        void Run(int separate)
        {
            if (separate == 0)
            {
                PackedFewRows(fused, gate, packed, scales ?? packed, y, m, n, k, words, align, up);
                return;
            }

            var hidden = Allocate(m * k, zeroed: false);
            try
            {
                GatedActivation(gate, up, hidden, m * k, activation);
                PackedFewRows((int)format * 4 + GemvPlain, hidden, packed, scales ?? packed, y, m, n, k, words, align);
            }
            finally
            {
                hidden.Release();
            }
        }

        // Before measuring: fused for int4, separate for the others.
        Run(Tune(new TuneKey(TuneOp.GatedActivation, fused, m, n, k), [0, 1], format == PackedFormat.Int4 ? 0 : 1, Run, cold: true));
        return true;
    }

    public override bool PackedMatMulAddRmsNorm(PackedFormat format, Storage x, Storage packed, Storage? scales, Storage y, int m, int n, int k,
        Storage residual, Storage sum, Storage gain, Storage normalized, float eps, float offset)
    {
        if (m > PtxKernels.GemvRows || k == 0)
        {
            return false;
        }

        int cpw = format.ValuesPerWord();
        PackedFewRows((int)format * 4 + GemvAddNorm, x, packed, scales ?? packed, y, m, n, k, (n + cpw - 1) / cpw, format.SplitAlignment(),
            tail: [P(residual), P(sum), P(gain), P(normalized), F(eps), F(offset)]);
        return true;
    }

    public override bool PackedMatMulMany(PackedFormat format, Storage x, int m, int k,
        ReadOnlySpan<(Storage Packed, Storage? Scales, Storage? Bias, Storage Output, int Columns)> products) =>
        PackedMany(format, x, m, k, products, -1, null);

    public override bool PackedMatMulGatedPair(PackedFormat format, int activation, Storage x, int m, int k,
        ReadOnlySpan<(Storage Packed, Storage? Scales, Storage? Bias, Storage Output, int Columns)> products, Storage hidden) =>
        products.Length == 2 && products[0].Columns == products[1].Columns && activation is >= 0 and <= 2
        && PackedMany(format, x, m, k, products, activation, hidden);

    // activation >= 0: the gate/up pair, with hidden = act(gate) · up written by the same launch.
    private bool PackedMany(PackedFormat format, Storage x, int m, int k,
        ReadOnlySpan<(Storage Packed, Storage? Scales, Storage? Bias, Storage Output, int Columns)> products, int activation, Storage? hidden)
    {
        if (m > PtxKernels.GemvRows && hidden is null)
        {
            return PackedManyLarge(format, x, m, k, products);
        }

        if (m > PtxKernels.GemvRows || k == 0 || products.Length is 0 or > 3)
        {
            return false;
        }

        int cpw = format.ValuesPerWord();
        int multiGemv = (int)format * 2 + (hidden is null ? 0 : 1);
        string kernel = GemvMultiKernels[multiGemv];
        int nmax = 0, totalBlocks = 0;
        foreach (var product in products)
        {
            nmax = Math.Max(nmax, product.Columns);
            totalBlocks += ((product.Columns + cpw - 1) / cpw + 31) / 32;
        }

        int columnBlocks = ((nmax + cpw - 1) / cpw + 31) / 32;
        int align = format.SplitAlignment();
        var counters = SplitCounters(columnBlocks * (products.Length + (hidden is null ? 0 : 1)));
        void Run(ReadOnlySpan<(Storage Packed, Storage? Scales, Storage? Bias, Storage Output, int Columns)> products, int wanted)
        {
            int chunk = ((k + wanted - 1) / wanted + align - 1) / align * align;
            int splits = (k + chunk - 1) / chunk;
            var part = splits > 1 ? Allocate(products.Length * splits * m * nmax, zeroed: false) : null;
            try
            {
                Span<ulong> args = stackalloc ulong[9 + 3 * 5 + 2];
                args[0] = P(x);
                args[1] = part is null ? P(products[0].Output) : P(part);
                args[2] = U(m);
                args[3] = U(k);
                args[4] = U(chunk);
                args[5] = U(splits);
                args[6] = P(counters);
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

                args[24] = U(activation);
                args[25] = hidden is null ? 0UL : P(hidden);
                Launch(K(kernel), (uint)columnBlocks, (uint)splits, (uint)products.Length, PtxKernels.Int8GemvThreads, 1, hidden is null ? args[..24] : args);
            }
            finally
            {
                part?.Release();
            }
        }

        int wanted;
        if (GemvSplits is int forced)
        {
            wanted = Math.Clamp(forced, 1, Math.Max(1, k / 16));
        }
        else
        {
            // Measured once per shape (every output, and the activation's, is written again by the run that follows).
            int biases = 0;
            for (int j = 0; j < products.Length; j++)
            {
                biases |= products[j].Bias is null ? 0 : 1 << j;
            }

            var name = new TuneKey(TuneOp.GemvMultiSplits, multiGemv, m, k, products[0].Columns, products.Length > 1 ? products[1].Columns : 0,
                products.Length > 2 ? products[2].Columns : 0, biases);
            int formula = GemvSplitCount(totalBlocks, k);
            var copy = TunedKnown(name) ? null : products.ToArray();
            wanted = copy is null ? Tune(name, [], formula, _ => { }) : Tune(name, GemvSplitCandidates(k, align), formula, c => Run(copy, c), cold: true);
        }

        Run(products, wanted);
        return true;
    }

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

    public override void Int8Dequantize(Storage q, Storage scales, Storage w, int k, int n)
    {
        int words = (n + 3) / 4;
        Launch1D(K("int8_dequant_f32"), k * words, P(q), P(scales), P(w), U(words), U(n), U(k * words));
    }

    public override void KeyValueWriteInt8(Storage source, Storage cache, Storage scales, Storage position, int heads, int steps, int capacity, int dim)
    {
        int n = heads * steps;
        Launch1D(K("kv_write_int8"), n, P(source), P(cache), P(scales), P(position), U(steps), U(capacity), U(dim), U((dim + 3) / 4), U(n));
    }

    public override void AttentionScoresInt8(Storage q, Storage cache, Storage scales, Storage y, int rows, int steps, int capacity, int dim)
    {
        int n = rows * steps * capacity;
        Launch1D(K("attn_scores_int8"), n, P(q), P(cache), P(scales), P(y), U(steps), U(capacity), U(dim), U((dim + 3) / 4), U(n));
    }

    public override void AttentionContextInt8(Storage weights, Storage cache, Storage scales, Storage y, int rows, int steps, int capacity, int dim)
    {
        int words = (dim + 3) / 4, n = rows * steps * words;
        Launch1D(K("attn_context_int8"), n, P(weights), P(cache), P(scales), P(y), U(steps), U(capacity), U(dim), U(words), U(n));
    }

    public override void RmsNorm(Storage x, Storage y, Storage inv, int rows, int cols, float eps) =>
        LaunchRows(K("rms_norm_f32"), rows, P(x), P(y), P(inv), U(cols), F(eps), U(rows));

    public override void RmsNormBackward(Storage dy, Storage y, Storage inv, Storage dx, int rows, int cols) =>
        LaunchRows(K("rms_norm_backward_f32"), rows, P(dy), P(y), P(inv), P(dx), U(cols), U(rows));

    public override void Rope(Storage x, Storage y, Storage cos, Storage sin, Storage positions, int rows, int heads, int steps, int dim, int half, bool interleaved, float sign)
    {
        int n = rows * half;
        Launch1D(K("rope_f32"), n, P(x), P(y), P(cos), P(sin), P(positions), U(heads), U(steps), U(dim), U(half), U(interleaved ? 1 : 0), F(sign), U(n));
    }

    public override void AttentionDecode(Storage q, Storage keys, Storage values, Storage position, Storage y, int heads, int rowsPerHead,
        int steps, int capacity, int dim, float scale, AttentionVariant variant = default)
    {
        if (dim > PtxKernels.DecodeMaxDim)
        {
            throw new NotSupportedException($"Decoding attention supports head sizes up to {PtxKernels.DecodeMaxDim} on CUDA.");
        }

        int rows = heads * rowsPerHead;
        DecodeSplit(WindowedTuning(0, variant, capacity), rows, capacity, dim, position, y, (splits, minChunk, part, counters, at) => Launch(K("attention_decode_f32"), (uint)rows, (uint)splits, 1, PtxKernels.RowThreads, 1,
            P(q), P(keys), P(values), at, P(y), P(part), P(counters), U(rowsPerHead), U(steps), U(capacity), U(dim), F(scale), U(minChunk),
            U(variant.Window), F(variant.Softcap), U(rows)));
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
    // doubling of the context counts alike. (Measured at a full cache alone, an RTX 3060 Laptop chose 64 splits, 2%
    // faster there than 8, and ran 200 positions in 36.8 µs against 12.0 with 8.) Candidates: 1, 2, 3, 4, 6, 8, ... up to one split
    // per 64 cached positions (at most 64). The formula, about five blocks per SM (--bench-gemv: 16 rows → 22 splits,
    // 4000 positions 75.9 → ~35 µs, 1000 positions 22 → ~15 µs), is the default until then and when nothing can be
    // measured.
    // A short filled length split as many ways reads a few positions per block and merges many parts: on an RTX 5070 Ti
    // 200 positions took 9.0 µs with the 24 splits measured at 4096, 7.6 with 16. So each block also takes at least
    // `minChunk` positions (the blocks past the filled length then add an empty part), measured per shape after the
    // splits over the same lengths; 1, the plain chunks, is the reference and stays unless another is faster. `launch(splits, minChunk, part, counters, position)` launches the kernel with that position address.
    private void DecodeSplit(int variant, int rows, int capacity, int dim, Storage position, Storage y, Action<int, int, Storage, Storage, ulong> launch)
    {
        var counters = SplitCounters(rows);
        void Run(int splits, int minChunk, ulong at)
        {
            if (splits == 1)
            {
                launch(1, 1, y, counters, at);
                return;
            }

            var part = Allocate(rows * splits * (dim + 2), zeroed: false);
            try
            {
                launch(splits, minChunk, part, counters, at);
            }
            finally
            {
                part.Release();
            }
        }

        int limit = Math.Clamp(capacity / 64, 1, 64);
        int splits, minChunk = 1;
        if (DecodeSplits is int forced)
        {
            splits = Math.Clamp(forced, 1, 64);
            minChunk = DecodeMinChunk ?? 1;
        }
        else
        {
            int formula = Math.Clamp((5 * Math.Max(1, _multiprocessors) + rows - 1) / rows, 1, Math.Min(32, limit));
            int[] candidates = SplitCounts(limit);
            if (!candidates.Contains(formula))
            {
                candidates = [.. candidates.Append(formula).Order()];
            }

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
                splits = Tune(new TuneKey(TuneOp.DecodeSplits, variant, rows, capacity, dim), candidates, formula, c =>
                {
                    for (int i = 0; i < lengths.Length; i++)
                    {
                        Run(c, 1, At(i));                               // writes y, which the launch below writes again
                    }
                }, part: (c, i) => Run(c, 1, At(i)), parts: lengths.Length);
                int[] chunks = splits > 1 ? DecodeMinChunks(capacity, splits) : [];
                minChunk = DecodeMinChunk ?? Tune(new TuneKey(TuneOp.DecodeMinChunk, variant, rows, capacity, dim, splits), chunks, 1, c =>
                {
                    for (int i = 0; i < lengths.Length; i++)
                    {
                        Run(splits, c, At(i));
                    }
                }, part: (c, i) => Run(splits, c, At(i)), parts: lengths.Length);
            }
            finally
            {
                positions?.Release();
            }
        }

        Run(splits, minChunk, P(position));
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
        var chunks = new List<int> { 1 };
        for (int c = 8; c * splits < capacity; c *= 2)
        {
            chunks.Add(c);
        }

        return [.. chunks];
    }

    public override void RmsNormAffine(Storage x, Storage gain, Storage y, int rows, int cols, float eps, float offset) =>
        LaunchRows(K("rms_norm_affine_f32"), rows, P(x), P(gain), P(y), U(cols), F(eps), F(offset), U(rows));

    public override void GatedActivation(Storage gate, Storage up, Storage y, int n, int kind) =>
        Launch1D(K("gated_act_f32"), n, P(gate), P(up), P(y), U(kind), U(n));

    public override void GatedActivationBackward(Storage gate, Storage up, Storage dy, Storage dgate, Storage dup, int n, int kind, int flags) =>
        Launch1D(K("gated_act_bwd_f32"), n, P(gate), P(up), P(dy), P(dgate), P(dup), U(kind), U(flags), U(n));

    public override void GatedActivationPacked(Storage gate, Storage up, Storage packedGate, Storage packedUp, Storage y, Storage packedY, int n, int kind, int flags) =>
        Launch1D(K("gated_act_bf16_f32"), (n + 1) / 2, P(gate), P(up), P(packedGate), P(packedUp), P(y), P(packedY), U(kind), U(flags), U(n), U((n + 1) / 2));

    public override void GatedActivationBackwardPacked(Storage packedGate, Storage packedUp, Storage dy, Storage dgate, Storage dup, int n, int kind, int flags) =>
        Launch1D(K("gated_act_bwd_bf16_f32"), (n + 1) / 2, P(packedGate), P(packedUp), P(dy), P(dgate), P(dup), U(kind), U(flags), U(n), U((n + 1) / 2));

    // Windows and soft-caps run on the float32 kernels (attention_flash_f32 and its backward); the tensor-core ones
    // compute plain causal attention only.
    public override void AttentionTiled(Storage q, Storage keys, Storage values, Storage position, Storage y, Storage? logSumExp, int heads,
        int rowsPerHead, int steps, int capacity, int dim, float scale, AttentionVariant variant = default)
    {
        if (variant.IsPlain && FlashTensorCore(dim) is { } tc)
        {
            Launch(tc[FlashNames(dim).Forward], (uint)((rowsPerHead + PtxKernels.FlashTensorRows - 1) / PtxKernels.FlashTensorRows), (uint)heads, 1, 128, 1,
                [P(q), P(keys), P(values), P(position), P(y), logSumExp is null ? 0UL : P(logSumExp),
                U(rowsPerHead), U(steps), U(capacity), F(scale * Log2E), .. ContiguousLayout(rowsPerHead, steps, capacity, dim)]);
            return;
        }

        if (dim > PtxKernels.FlashMaxDim)
        {
            base.AttentionTiled(q, keys, values, position, y, logSumExp, heads, rowsPerHead, steps, capacity, dim, scale, variant);
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

    // Strides for [batch, steps, *] rows (see Backend.AttentionStrided).
    private static ulong[] RowLayout(int kvHeads, int group, int steps, int dim, int qRow, int kRow, int yRow) =>
        [U(kvHeads), U(steps * qRow), U(group * dim), U(dim), U(qRow), U(steps * yRow), U(group * dim), U(dim), U(yRow),
            U(steps * kRow), U(dim), U(kRow), 0UL, 0UL, 0UL];

    private Storage ZeroPosition => _zeroPosition ??= Allocate(1, zeroed: true);

    private Storage? _zeroPosition;

    public override bool AttentionStrided(Storage q, long qOffset, Storage k, long kOffset, Storage v, long vOffset, int qRow, int kRow,
        Storage y, Storage? logSumExp, int batch, int kvHeads, int group, int steps, int dim, float scale)
    {
        if (!PtxKernels.FlashTensorDim(dim) || !FlashKernelsLoaded(dim))
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

    public override bool AttentionStridedBackward(Storage q, long qOffset, Storage k, long kOffset, Storage v, long vOffset, int qRow, int kRow,
        Storage y, Storage logSumExp, Storage dOutput, Storage dq, long dqOffset, Storage dk, long dkOffset, Storage dv, long dvOffset,
        int batch, int kvHeads, int group, int steps, int dim, float scale)
    {
        if (!PtxKernels.FlashTensorDim(dim) || !FlashKernelsLoaded(dim))
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

    public override void SumColumns(Storage x, long offset, int ld, Storage y, int rows, int cols)
    {
        const int Chunk = 64;
        Launch(K("sum_cols_strided_f32"), (uint)((cols + _shapes.BlockSize - 1) / _shapes.BlockSize), (uint)((rows + Chunk - 1) / Chunk), 1, (uint)_shapes.BlockSize, 1,
            P(x) + (ulong)offset * 4, P(y), U(rows), U(cols), U(ld), U(Chunk));
    }

    // The tensor-core flash kernels when MixedPrecision asks for bfloat16 and the head size and GPU allow them.
    private Dictionary<string, IntPtr>? FlashTensorCore(int dim) =>
        MixedPrecision.UsesTensorCores && PtxKernels.FlashTensorDim(dim) && FlashKernelsLoaded(dim) ? _tensorCoreAny : null;

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

    public override void AttentionTiledBackward(Storage q, Storage keys, Storage values, Storage output, Storage logSumExp, Storage dOutput,
        Storage dq, Storage dkeys, Storage dvalues, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale, AttentionVariant variant = default)
    {
        int rows = heads * rowsPerHead;
        var delta = Allocate(rows, zeroed: false);
        try
        {
            Launch1D(K("attn_bwd_d_f32"), rows, P(output), P(dOutput), P(delta), U(dim), U(rows));
            if (variant.IsPlain && FlashTensorCore(dim) is { } tc)
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

    // Packed sequences and rows of different lengths run on the tensor-core flash kernels, which compute plain causal
    // attention only: a window or a cap is refused (fine-tuning then pads, batches decode one by one).
    public override bool SupportsSegmentedAttention(int dim, AttentionVariant variant = default) => variant.IsPlain && FlashTensorCore(dim) is not null;

    public override bool AttentionRows(Storage q, Storage keys, Storage values, Storage position, Storage y, Storage starts, int heads, int headsPerRow,
        int rowsPerHead, int steps, int capacity, int dim, float scale, AttentionVariant variant = default)
    {
        if (!variant.IsPlain || FlashTensorCore(dim) is not { } tc)
        {
            return false;
        }

        Launch(tc[FlashNames(dim).Forward], (uint)((rowsPerHead + PtxKernels.FlashTensorRows - 1) / PtxKernels.FlashTensorRows), (uint)heads, 1, 128, 1,
            [P(q), P(keys), P(values), P(position), P(y), 0UL, U(rowsPerHead), U(steps), U(capacity), F(scale * Log2E),
            .. ContiguousLayout(rowsPerHead, steps, capacity, dim, starts, starts, headsPerRow)]);
        return true;
    }

    public override bool AttentionSegmented(Storage q, Storage keys, Storage values, Storage y, Storage? logSumExp, Storage starts, Storage ends,
        int heads, int headsPerRow, int rowsPerHead, int steps, int dim, float scale, AttentionVariant variant = default)
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

    public override bool AttentionSegmentedBackward(Storage q, Storage keys, Storage values, Storage output, Storage logSumExp, Storage dOutput,
        Storage dq, Storage dkeys, Storage dvalues, Storage starts, Storage ends, int heads, int headsPerRow, int rowsPerHead, int steps, int dim, float scale,
        AttentionVariant variant = default)
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

    public override void AttentionInt8(Storage q, Storage keys, Storage values, Storage keyScales, Storage valueScales, Storage position,
        Storage y, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale, bool tiled, AttentionVariant variant = default)
    {
        int words = (dim + 3) / 4;
        if (tiled && dim <= PtxKernels.FlashMaxDim)
        {
            Launch(K("attention_flash_int8"), (uint)((rowsPerHead + PtxKernels.FlashTile - 1) / PtxKernels.FlashTile), (uint)heads, 1, 128, 1,
                P(q), P(keys), P(values), P(position), P(y), 0UL, U(rowsPerHead), U(steps), U(capacity), U(dim), F(scale),
                P(keyScales), P(valueScales), U(words), U(variant.Window), F(variant.Softcap));
            return;
        }

        if (dim > PtxKernels.DecodeMaxDim)
        {
            throw new NotSupportedException($"Int8 cache attention supports head sizes up to {PtxKernels.DecodeMaxDim} on CUDA.");
        }

        int rows = heads * rowsPerHead;
        DecodeSplit(WindowedTuning(1, variant, capacity), rows, capacity, dim, position, y, (splits, minChunk, part, counters, at) => Launch(K("attention_decode_int8"), (uint)rows, (uint)splits, 1, PtxKernels.RowThreads, 1,
            P(q), P(keys), P(values), P(keyScales), P(valueScales), at, P(y), P(part), P(counters),
            U(rowsPerHead), U(steps), U(capacity), U(dim), F(scale), U(words), U(minChunk), U(variant.Window), F(variant.Softcap), U(rows)));
    }

    public override void SoftmaxCrossEntropyRows(Storage logits, Storage targets, Storage weights, Storage losses, int rows, int vocabulary, float scale) =>
        LaunchRows(K("softmax_ce_rows_f32"), rows, PtxKernels.RowThreads * 4, P(logits), P(targets), P(weights), P(losses), U(vocabulary), F(scale), U(rows));

    public override void AttentionBFloat16(Storage q, Storage keys, Storage values, Storage position, Storage y, int heads, int rowsPerHead,
        int steps, int capacity, int dim, float scale, bool tiled, AttentionVariant variant = default)
    {
        int words = (dim + 1) / 2;
        if (tiled && dim <= PtxKernels.FlashMaxDim)
        {
            Launch(K("attention_flash_bf16"), (uint)((rowsPerHead + PtxKernels.FlashTile - 1) / PtxKernels.FlashTile), (uint)heads, 1, 128, 1,
                P(q), P(keys), P(values), P(position), P(y), 0UL, U(rowsPerHead), U(steps), U(capacity), U(dim), F(scale), U(words),
                U(variant.Window), F(variant.Softcap));
            return;
        }

        if (dim > PtxKernels.DecodeMaxDim)
        {
            throw new NotSupportedException($"bfloat16 cache attention supports head sizes up to {PtxKernels.DecodeMaxDim} on CUDA.");
        }

        int rows = heads * rowsPerHead;
        DecodeSplit(WindowedTuning(2, variant, capacity), rows, capacity, dim, position, y, (splits, minChunk, part, counters, at) => Launch(K("attention_decode_bf16"), (uint)rows, (uint)splits, 1, PtxKernels.RowThreads, 1,
            P(q), P(keys), P(values), at, P(y), P(part), P(counters), U(rowsPerHead), U(steps), U(capacity), U(dim), F(scale), U(words), U(minChunk),
            U(variant.Window), F(variant.Softcap), U(rows)));

    }

    public override void KeyValueWriteBFloat16(Storage source, Storage cache, Storage position, int heads, int steps, int capacity, int dim)
    {
        int words = (dim + 1) / 2, n = heads * steps * words;
        Launch1D(K("kv_write_bf16"), n, P(source), P(cache), P(position), U(steps), U(capacity), U(dim), U(words), U(n));
    }

    public override void AddRmsNormAffine(Storage a, Storage b, Storage sum, Storage gain, Storage y, int rows, int cols, float eps, float offset) =>
        LaunchRows(K("add_rms_norm_affine_f32"), rows, P(a), P(b), P(sum), P(gain), P(y), U(cols), F(eps), F(offset), U(rows));

    public override void RmsNormRope(Storage x, Storage gain, Storage cos, Storage sin, Storage positions, Storage y, int rows, int cols,
        float eps, float offset, int heads, int steps, int half, bool interleaved) =>
        LaunchRows(K("rms_norm_rope_f32"), rows, P(x), P(gain), P(cos), P(sin), P(positions), P(y),
            U(cols), F(eps), F(offset), U(heads), U(steps), U(half), U(interleaved ? 1 : 0), U(rows));

    public override void RmsNormRopePair(Storage x, Storage gain, Storage y, int rows1, float eps, float offset, int heads,
        Storage x2, Storage gain2, Storage y2, int rows2, float eps2, float offset2, int heads2,
        Storage cos, Storage sin, Storage positions, int cols, int steps, int half, bool interleaved) =>
        LaunchRows(K("rms_norm_rope2_f32"), rows1 + rows2, P(x), P(gain), P(cos), P(sin), P(positions), P(y), P(x2), P(gain2), P(y2),
            U(cols), F(eps), F(offset), U(heads), U(steps), U(half), U(interleaved ? 1 : 0), U(rows1), F(eps2), F(offset2), U(heads2),
            U(rows1 + rows2));

    public override bool NormRopeHeads(Storage q, Storage k, Storage v, int batch, int steps, int heads, int kvHeads, int cols,
        Storage? gainQ, float epsQ, float offsetQ, Storage? gainK, float epsK, float offsetK, Storage? cos, Storage? sin, Storage? positions,
        int half, bool interleaved, Storage yq, Storage yk, Storage yv, Storage? position, int capacity, int stride, bool bfloat16)
    {
        int rows1 = batch * steps * heads, rows2 = batch * steps * kvHeads;
        int flags = (gainQ is not null ? 1 : 0) | (bfloat16 ? 2 : 0) | (position is not null ? 4 : 0);
        static ulong Optional(Storage? s) => s is null ? 0UL : P(s);
        LaunchRows(K("norm_rope_heads_f32"), rows1 + 2 * rows2, P(q), Optional(gainQ), P(k), Optional(gainK), P(v), Optional(cos), Optional(sin),
            Optional(positions), Optional(position), P(yq), P(yk), P(yv), U(cols), F(epsQ), F(offsetQ), U(heads), F(epsK), F(offsetK), U(kvHeads),
            U(steps), U(cos is null ? 0 : half), U(interleaved ? 1 : 0), U(rows1), U(rows2), U(flags), U(capacity), U(stride), U(rows1 + 2 * rows2));
        return true;
    }
}
