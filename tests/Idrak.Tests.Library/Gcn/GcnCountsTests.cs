// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;
using Idrak.Gpu.Abstractions;
using Idrak.Gpu.Hip;
using Idrak.Tests.Gcn;
using GcnWeights = Idrak.Gpu.Hip.GcnKernels.PackedWeights;
using Variant = Idrak.Gpu.Hip.GcnKernels.PackedGemvVariant;

// The interpreter's counts of the kernels plan 35's G6b made faster (rules 80-83): the instructions the waves executed,
// and of them the vector and scalar memory instructions, the LDS instructions, the waits and the barriers, per element of
// the work, and the launches, printed so a change's evidence can be read where no device is (the device's own timing is
// --bench-hip's). The work is the same for both forms of a change, so the lines compare directly.
internal abstract partial class Tests
{
    // One printed line: the launches and the counts per element of `elements`.
    private static void CountsLine(string what, double elements, params GcnRunResult[] runs)
    {
        long instructions = 0;
        var counts = default(GcnCounts);
        foreach (var run in runs)
        {
            instructions += run.Instructions;
            counts += run.Counts;
        }

        Console.WriteLine(FormattableString.Invariant(
            $"    {what}: {runs.Length} launch(es); per element {instructions / elements:G4} instructions, {counts.VectorMemory / elements:G4} vector memory, {counts.ScalarMemory / elements:G4} scalar memory, {counts.Lds / elements:G4} LDS, {counts.Waits / elements:G4} waits, {counts.Barriers / elements:G4} barriers"));
    }

    internal static void GcnTrainingCounts(Device device)
    {
        _ = device;
        var isa = TrainingIsas[0];
        var random = new Random(70);

        // The attention backward passes over a causal cache: 2 heads, 200 query rows and positions, head 64.
        const int Heads = 2, Steps = 200, Dim = 64;
        const float Scale = 0.125f;
        int rows = Heads * Steps;
        float[] q = TrainingRandom(random, rows * Dim, -1, 1), k = TrainingRandom(random, Heads * Steps * Dim, -1, 1), v = TrainingRandom(random, Heads * Steps * Dim, -1, 1);
        float[][] forward = RowsOnCpu([q, k, v, [0f], new float[rows * Dim], new float[rows]],
            (b, s) => b.AttentionTiledKernel(s[0], s[1], s[2], s[3], s[4], s[5], Heads, Steps, Steps, Steps, Dim, Scale));
        float[] dout = TrainingRandom(random, rows * Dim, -1, 1);
        double work = (double)Heads * Steps * Steps * Dim;
        var runs = new GcnRunResult[2];
        float[][] transposed = [HeadsTransposed(q, Heads, Steps, Dim), HeadsTransposed(dout, Heads, Steps, Dim), HeadsTransposed(k, Heads, Steps, Dim), HeadsTransposed(v, Heads, Steps, Dim)];
        float[][] tiled = RowsOnGcn([q, k, v, forward[4], forward[5], dout, new float[rows * Dim], new float[Heads * Steps * Dim], new float[Heads * Steps * Dim], new float[rows],
            .. transposed], (m, a) =>
        {
            TrainingDispatch("attn_bwd_d_f32", isa, m, (rows, 1), 1, a[3], a[5], a[9], Dim, (ulong)rows);
            ulong[] Args(int n) => [a[0], a[1], a[2], a[5], a[4], a[9], a[6], a[7], a[8], a[10], a[11], a[12], a[13], Steps, Steps, Steps, Dim, Bits(Scale), 0, 0, (ulong)n];
            runs[0] = TrainingDispatch("attn_bwd_kv_f32", isa, m, (Heads * Steps, 1), 1, Args(Heads * Steps));
            runs[1] = TrainingDispatch("attn_bwd_q_f32", isa, m, (rows, 1), 1, Args(rows));
        });
        CountsLine($"attn_bwd_kv_f32, causal, {Heads} heads x {Steps} rows x {Steps} positions, head {Dim} ({isa}; per row x position x dimension)", work, runs[0]);
        CountsLine($"attn_bwd_q_f32, causal, {Heads} heads x {Steps} rows x {Steps} positions, head {Dim} ({isa}; per row x position x dimension)", work, runs[1]);

        // Over key ranges: the same shape, causal ranges in one table.
        float[] starts = new float[Steps], ends = [.. Enumerable.Range(1, Steps).Select(i => (float)i)];
        var spans = new GcnRunResult[1];
        float[][] spanned = RowsOnGcn([q, k, v, starts, ends, dout, forward[5], new float[rows], new float[rows * Dim], new float[Heads * Steps * Dim], new float[Heads * Steps * Dim],
            .. transposed], (m, a) =>
        {
            ulong[] Args(int n) => [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], Steps, Steps, Dim, Bits(Scale), 1, Heads, 0,
                Heads, (ulong)n];
            spans[0] = TrainingDispatch("attn_spans_bwd_kv_f32", isa, m, (Heads * Steps, 1), 1, Args(Heads * Steps));
        });
        CountsLine($"attn_spans_bwd_kv_f32, causal ranges, {Heads} heads x {Steps} rows x {Steps} positions, head {Dim} ({isa}; per row x position x dimension)", work, spans[0]);

        // The column sums: 64 rows of 1,000.
        const int SumRows = 64, SumCols = 1000;
        var sums = new GcnRunResult[2];
        float[][] summed = RowsOnGcn([TrainingRandom(random, SumRows * SumCols, -1, 1), new float[SumCols], new float[SumCols]], (m, a) =>
        {
            sums[0] = TrainingDispatch("sum_rows_f32", isa, m, (4, 1), 1, a[0], a[1], SumRows, SumCols);
            sums[1] = TrainingDispatch("sum_cols_strided_f32", isa, m, (4, 1), 1, a[0], a[2], a[2], SumRows, SumCols, SumCols, SumRows);
        });
        CountsLine($"sum_rows_f32, {SumRows} x {SumCols} ({isa}; per element)", SumRows * SumCols, sums[0]);
        CountsLine($"sum_cols_strided_f32, {SumRows} x {SumCols}, one chunk ({isa}; per element)", SumRows * SumCols, sums[1]);

        // The outputs' bits, against the kernels before G6b's (the same inputs, run on the build before it): the dot
        // products' order is kept and the skipped rows' terms were exact zeros, the sums' rows are added in order.
        (string What, float[] Values, string Before)[] outputs =
        [
            ("attn_bwd_q_f32's dq", tiled[6], TiledDqBefore), ("attn_bwd_kv_f32's dk", tiled[7], TiledDkBefore), ("attn_bwd_kv_f32's dv", tiled[8], TiledDvBefore),
            ("attn_spans_bwd_kv_f32's dk", spanned[9], SpansDkBefore), ("attn_spans_bwd_kv_f32's dv", spanned[10], SpansDvBefore),
            ("sum_rows_f32's sums", summed[1], SumRowsBefore), ("sum_cols_strided_f32's sums", summed[2], SumColsBefore),
        ];
        foreach (var (what, values, before) in outputs)
        {
            string hash = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(values.AsSpan())))[..16];
            Console.WriteLine($"    {what}: bits {hash}");
            Check(hash == before, $"{what}: bits {hash}, before G6b {before} (the same bits expected)");
        }
    }

    internal static void GcnDecodingCounts(Device device)
    {
        _ = device;
        var isa = ProductIsas[0];
        var random = new Random(71);

        // A split int8 product (1 row, 1,024 x 1,024, 8 splits), alone and with the residual and norm.
        const int N = 1024, K = 1024, Chunk = 128;
        int splits = K / Chunk, words = N / 4, blocks = (words + GcnKernels.PackedGemvWords - 1) / GcnKernels.PackedGemvWords;
        var (packed, scales) = ProductsWeights(GcnWeights.Int8, random, K, N);
        float[] x = RowsRandom(random, K, 1f), residual = RowsRandom(random, N, 1f), gain = RowsRandom(random, N, 0.5f);
        float[][] Product() => [x, packed, scales, new float[N], new float[splits * N], new float[blocks + 1], residual, gain, new float[N], new float[N]];
        GcnRunResult[] two = new GcnRunResult[2], one = new GcnRunResult[1];
        string plain = GcnKernels.PackedGemvKernel(GcnWeights.Int8, 1), fused = GcnKernels.FusedGemvKernel(GcnWeights.Int8, Variant.AddNorm, 1);
        ulong[] ProductArgs(ulong[] a, string kernel, ulong counters) =>
            [a[0], a[1], a[2], a[3], a[4], 1, N, K, (ulong)words, Chunk, (ulong)splits, counters];
        _ = RowsOnGcn(Product(), (m, a) =>
        {
            two[0] = DecodingDispatch(plain, isa, m, (blocks, splits, 1), 1, ProductArgs(a, plain, 0));
            two[1] = DecodingDispatch("int8_gemv_finish_f32", isa, m, (DecodingBlocks(N, true), 1, 1), 1, a[4], a[2], a[3], N, N, (ulong)splits, N);
        });
        _ = RowsOnGcn(Product(), (m, a) => one[0] = DecodingDispatch(plain, isa, m, (blocks, splits, 1), 1, ProductArgs(a, plain, a[5])));
        CountsLine($"{plain} 1x{N}x{K}, {splits} splits + int8_gemv_finish_f32 ({isa}; per output)", N, two);
        CountsLine($"{plain} 1x{N}x{K}, {splits} splits, the last blocks add them ({isa}; per output)", N, one);
        _ = RowsOnGcn(Product(), (m, a) =>
        {
            two[0] = DecodingDispatch(plain, isa, m, (blocks, splits, 1), 1, ProductArgs(a, plain, 0));
            two[1] = DecodingDispatch("gemv_addnorm_f32", isa, m, (1, 1, 1), 1, a[4], a[2], a[3], a[6], a[8], a[7], a[9], 0UL, N, (ulong)splits, N, Bits(1e-6f), 0, Bits(1e-6f), 0, 0, 1);
        });
        _ = RowsOnGcn(Product(), (m, a) => one[0] = DecodingDispatch(fused, isa, m, (blocks, splits, 1), 1,
            [.. ProductArgs(a, fused, a[5]), a[6], a[8], a[7], a[9], 0UL, Bits(1e-6f), 0, Bits(1e-6f), 0, 0]));
        CountsLine($"{plain} + gemv_addnorm_f32 1x{N}x{K}, {splits} splits ({isa}; per output)", N, two);
        CountsLine($"{fused} 1x{N}x{K}, {splits} splits, the last block adds and normalizes ({isa}; per output)", N, one);

        // A bfloat16 gate/up pair (1 row, 2 x 512 of 512, 4 splits).
        const int Pair = 512, PairK = 512, PairChunk = 128;
        int pairSplits = PairK / PairChunk, pairBlocks = (Pair / 2 + GcnKernels.PackedGemvWords - 1) / GcnKernels.PackedGemvWords;
        var (gateWeights, _) = ProductsWeights(GcnWeights.BFloat16, random, PairK, Pair);
        var (upWeights, _) = ProductsWeights(GcnWeights.BFloat16, random, PairK, Pair);
        float[] px = RowsRandom(random, PairK, 1f);
        string multi = GcnKernels.FusedGemvKernel(GcnWeights.BFloat16, Variant.Multi, 1);
        float[][] Pairs() => [px, gateWeights, upWeights, new float[Pair], new float[Pair], new float[Pair], new float[2 * pairSplits * Pair], new float[3 * pairBlocks]];
        ulong[] MultiArgs(ulong[] a, bool lastBlock)
        {
            var values = new List<ulong> { a[0], a[6], 1, PairK, PairChunk, (ulong)pairSplits, lastBlock ? a[7] : 0UL, (ulong)pairBlocks, Pair };
            for (int j = 0; j < 3; j++)
            {
                values.AddRange([a[1 + Math.Min(j, 1)], 0UL, a[3 + Math.Min(j, 1)], 0UL, j < 2 ? Pair : 0UL]);
            }

            values.AddRange([lastBlock ? 1UL : 0UL, lastBlock ? a[5] : 0UL]);
            return [.. values];
        }

        _ = RowsOnGcn(Pairs(), (m, a) =>
        {
            two[0] = DecodingDispatch(multi, isa, m, (pairBlocks, pairSplits, 2), 1, MultiArgs(a, false));
            var finish = new List<ulong> { a[6], a[5], (ulong)pairSplits, 1, Pair, 1 };
            for (int j = 0; j < 3; j++)
            {
                finish.AddRange([0UL, a[3 + Math.Min(j, 1)], 0UL, j < 2 ? Pair : 0UL]);
            }

            two[1] = DecodingDispatch("gemv_multi_finish_f32", isa, m, (DecodingBlocks(Pair, true), 1, 1), 1, [.. finish]);
        });
        _ = RowsOnGcn(Pairs(), (m, a) => one[0] = DecodingDispatch(multi, isa, m, (pairBlocks, pairSplits, 2), 1, MultiArgs(a, true)));
        CountsLine($"{multi} gate/up pair 1x2x{Pair}x{PairK}, {pairSplits} splits + gemv_multi_finish_f32 ({isa}; per output)", 2 * Pair, two);
        CountsLine($"{multi} gate/up pair 1x2x{Pair}x{PairK}, {pairSplits} splits, the last blocks ({isa}; per output)", 2 * Pair, one);

        // Decoding attention split over a cache: 8 heads of 64, 2,048 positions in 4 splits.
        const int Heads = 8, Dim = 64, Capacity = 2048, AttentionChunk = 512;
        int attentionSplits = Capacity / AttentionChunk;
        float[][] Attention() =>
        [
            RowsRandom(random, Heads * Dim, 1f), RowsRandom(random, Heads * Capacity * Dim, 1f), RowsRandom(random, Heads * Capacity * Dim, 1f), [Capacity - 1],
            new float[Heads * Dim], new float[Heads * attentionSplits * (Dim + 2)], new float[Heads],
        ];
        ulong[] DecodeArgs(ulong[] a, ulong counters) =>
            [a[0], a[1], a[2], a[3], a[4], 0UL, a[5], counters, 1, 1, Capacity, Dim, Bits(0.125f), 0, 0, AttentionChunk, (ulong)attentionSplits, Heads];
        _ = RowsOnGcn(Attention(), (m, a) =>
        {
            two[0] = DecodingDispatch("attention_decode_f32", isa, m, (Heads, attentionSplits, 1), 1, DecodeArgs(a, 0));
            two[1] = DecodingDispatch("attention_merge_f32", isa, m, (Heads, 1, 1), 1, a[5], a[4], 0UL, (ulong)attentionSplits, Dim, 1, Heads, 0, 0, Dim, Heads);
        });
        _ = RowsOnGcn(Attention(), (m, a) => one[0] = DecodingDispatch("attention_decode_f32", isa, m, (Heads, attentionSplits, 1), 1, DecodeArgs(a, a[6])));
        double positions = (double)Heads * Capacity * Dim;
        CountsLine($"attention_decode_f32 {Heads} rows x {Capacity} x {Dim}, {attentionSplits} splits + attention_merge_f32 ({isa}; per row x position x dimension)", positions, two);
        CountsLine($"attention_decode_f32 {Heads} rows x {Capacity} x {Dim}, {attentionSplits} splits, the last blocks merge ({isa}; per row x position x dimension)", positions, one);

        // The tiled form: 2 heads x 128 rows over themselves (causal), head 64, a block a row against 8 rows a block.
        const int TiledHeads = 2, TiledRows = 128, TiledDim = 64;
        float[][] Tiled() =>
        [
            RowsRandom(random, TiledHeads * TiledRows * TiledDim, 1f), RowsRandom(random, TiledHeads * TiledRows * TiledDim, 1f),
            RowsRandom(random, TiledHeads * TiledRows * TiledDim, 1f), [0f], new float[TiledHeads * TiledRows * TiledDim], new float[TiledHeads * TiledRows],
        ];
        _ = RowsOnGcn(Tiled(), (m, a) => one[0] = DecodingDispatch("attention_decode_f32", isa, m, (TiledHeads * TiledRows, 1, 1), 1, a[0], a[1], a[2], a[3], a[4], a[5], 0UL, 0UL,
            TiledRows, TiledRows, TiledRows, TiledDim, Bits(0.125f), 0, 0, TiledRows, 1, TiledHeads * TiledRows));
        double causal = (double)TiledHeads * TiledRows * (TiledRows + 1) / 2 * TiledDim;
        CountsLine($"attention_decode_f32 as the tiled form, {TiledHeads} heads x {TiledRows} rows (causal), head {TiledDim}, a block a row ({isa}; per row x seen position x dimension)", causal, one);
        foreach (int perBlock in GcnKernels.AttentionTiledRows)
        {
            string kernel = GcnKernels.AttentionTiledKernel(perBlock);
            int bph = TiledRows / perBlock;
            _ = RowsOnGcn(Tiled(), (m, a) => one[0] = DecodingDispatch(kernel, isa, m, (TiledHeads * bph, 1, 1), 1, a[0], a[1], a[2], a[3], a[4], a[5], TiledRows, TiledRows,
                TiledRows, TiledDim, Bits(0.125f), 0, 0, (ulong)bph, (ulong)(TiledHeads * bph)));
            CountsLine($"{kernel}, {TiledHeads} heads x {TiledRows} rows (causal), head {TiledDim}, {perBlock} rows a block ({isa}; per row x seen position x dimension)", causal, one);
        }

        // Greedy sampling of 2 rows of 9,000 tokens (5 candidate blocks a row).
        const int Rows = 2, Vocabulary = 9000;
        int candidateBlocks = (Vocabulary + GcnKernels.CandidateSlice - 1) / GcnKernels.CandidateSlice, slots = candidateBlocks * GcnKernels.CandidateSlots;
        var settings = GcnKernels.SamplerSettings.Of(Vocabulary, Vocabulary, 0, 1f, 1, 1f, 0f, 9u, Rows);
        float[][] Sampling() =>
        [
            RowsRandom(random, Rows * Vocabulary, 6f), new float[Rows], new float[Rows * 13], [0f], new float[Rows * slots], new float[Rows * slots], new float[Rows * candidateBlocks],
            new float[Rows * candidateBlocks * 3], new float[Rows],
        ];
        _ = RowsOnGcn(Sampling(), (m, a) =>
        {
            two[0] = DecodingPacked("topk_greedy_f32", isa, m, (Rows * candidateBlocks, 1, 1), 1,
                bytes => GcnKernels.PackCandidates(bytes, true, a[0], a[4], a[5], a[6], a[7], settings, (uint)candidateBlocks, a[1], a[2], a[3], 0UL));
            two[1] = DecodingPacked("sample_candidates_greedy_f32", isa, m, (Rows, 1, 1), 1,
                bytes => GcnKernels.PackSampler(bytes, GcnKernels.SamplerForm.Greedy, a[0], a[1], a[2], a[3], settings, a[4], a[5], a[6], a[7], (uint)slots));
        });
        _ = RowsOnGcn(Sampling(), (m, a) => one[0] = DecodingPacked("topk_greedy_f32", isa, m, (Rows * candidateBlocks, 1, 1), 1,
            bytes => GcnKernels.PackCandidates(bytes, true, a[0], a[4], a[5], a[6], a[7], settings, (uint)candidateBlocks, a[1], a[2], a[3], a[8])));
        CountsLine($"topk_greedy_f32 + sample_candidates_greedy_f32, {Rows} rows of {Vocabulary} ({isa}; per score)", Rows * Vocabulary, two);
        CountsLine($"topk_greedy_f32, its last block of a row samples it, {Rows} rows of {Vocabulary} ({isa}; per score)", Rows * Vocabulary, one);
    }

    // The outputs' hashes (SHA-256, first 8 bytes) of the kernels before G6b on the inputs above (main 52bf7e3, gfx90c).
    private const string TiledDqBefore = "5BFEC5852875DA06", TiledDkBefore = "FDE698702F1A9276", TiledDvBefore = "469EE866F5BE35EE", SpansDkBefore = "854FD126B19FA304",
        SpansDvBefore = "469EE866F5BE35EE", SumRowsBefore = "97D0E68B9FD831CB", SumColsBefore = "97D0E68B9FD831CB";
}
