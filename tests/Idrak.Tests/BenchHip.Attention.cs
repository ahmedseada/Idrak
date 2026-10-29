// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Gpu.Hip;

// --bench-hip attention and shapes (plan 35, G4b): HIP's attention and shape operations on their GCN kernels, each timed
// cold (its first call) apart from warm (BenchOperation), with GB/s and GFLOP/s from the tables of
// GcnKernels.Attention.cs and GcnKernels.Shapes.cs (HipBackend.Cost), each section ending with the host fallbacks it took
// (none expected). Sizes follow the large buffer (from the memory the device reports), never a card's: a cache of up to
// 4,096 positions of 32 heads of 128 (one row a head for decoding), a prompt of up to 512 rows for the tiled form and the
// key ranges, the shapes' buffers within a quarter of the large one.
internal abstract partial class DesktopTests
{
    // Storages on the device for one section, released at its end.
    private sealed class BenchStorages(HipBackend hip, Random random) : IDisposable
    {
        private readonly List<Storage> _owned = [];

        public Storage Random(int count, float low = -1f, float high = 1f)
        {
            var values = new float[count];
            for (int i = 0; i < count; i++)
            {
                values[i] = low + random.NextSingle() * (high - low);
            }

            return Of(values);
        }

        public Storage Of(float[] values)
        {
            var s = hip.Allocate(values.Length, zeroed: false);
            _owned.Add(s);
            hip.Upload(values, s);
            return s;
        }

        public void Dispose()
        {
            foreach (var s in _owned)
            {
                s.Release();
            }
        }
    }

    private static void BenchHipAttention(HipBench b)
    {
        if (!BenchKernelSet(b))
        {
            return;
        }

        var hip = b.Hip;
        long before = Idrak.Abstraction.Operations.Kernels.HostCalls(hip);
        const int Heads = 32, KvHeads = 8, Dim = 128;
        const float Scale = 0.088f;
        int capacity = (int)Math.Clamp(b.LargeElements / (8L * Heads * Dim), 256, 4096);
        int prompt = Math.Min(capacity, 512);
        using (var on = new BenchStorages(hip, new Random(7)))
        {
            Storage q = on.Random(Heads * Dim), keys = on.Random(Heads * capacity * Dim), values = on.Random(Heads * capacity * Dim), y = on.Random(Heads * Dim);
            Storage last = on.Of([capacity - 1]), start = on.Of([0f]);
            var (splits, chunk) = HipBackend.AttentionSplits(Heads, capacity, Dim, HipBackend.AttentionSplitPositions(Environment.GetEnvironmentVariable(HipBackend.AttentionSplitVariable)));
            b.Row("  (splits a decoding row takes)", string.Create(CultureInfo.InvariantCulture,
                $"{splits} of {chunk} positions for {Heads} rows over {capacity} ({HipBackend.AttentionSplitVariable}), merged in split order by each row's last split block, or by attention_merge_f32 ({HipBackend.FinishVariable})"));
            string shape = string.Create(CultureInfo.InvariantCulture, $"{Heads} heads x {capacity} positions, head {Dim}");
            long decoding = (long)Heads * capacity * Dim;
            BenchFinishes(form => BenchOperation(b, $"decoding, float32 cache, {shape}{form}", decoding, HipBackend.Cost("attention_decode_f32"),
                () => hip.AttentionDecode(q, keys, values, last, y, Heads, 1, 1, capacity, Dim, Scale)), both: splits > 1);

            // The packed caches written from the float32 one on the device (every position, as a prompt would).
            Storage keys16 = on.Random(Heads * capacity * (Dim / 2)), values16 = on.Random(Heads * capacity * (Dim / 2));
            hip.KeyValueWriteBFloat16(keys, keys16, start, Heads, capacity, capacity, Dim);
            hip.KeyValueWriteBFloat16(values, values16, start, Heads, capacity, capacity, Dim);
            BenchFinishes(form => BenchOperation(b, $"decoding, bfloat16 cache, {shape}{form}", decoding, HipBackend.Cost("attention_decode_bf16"),
                () => hip.AttentionBFloat16(q, keys16, values16, last, y, Heads, 1, 1, capacity, Dim, Scale, tiled: false)), both: splits > 1);
            Storage keys8 = on.Random(Heads * capacity * (Dim / 4)), values8 = on.Random(Heads * capacity * (Dim / 4));
            Storage keyScales = on.Random(Heads * capacity), valueScales = on.Random(Heads * capacity);
            hip.KeyValueWriteInt8(keys, keys8, keyScales, start, Heads, capacity, capacity, Dim);
            hip.KeyValueWriteInt8(values, values8, valueScales, start, Heads, capacity, capacity, Dim);
            BenchFinishes(form => BenchOperation(b, $"decoding, int8 cache, {shape}{form}", decoding, HipBackend.Cost("attention_decode_int8"),
                () => hip.AttentionInt8(q, keys8, values8, keyScales, valueScales, last, y, Heads, 1, 1, capacity, Dim, Scale, tiled: false)), both: splits > 1);

            // A prompt over itself: each row sees the positions up to its own (causal), with the log-sum-exp.
            Storage qp = on.Random(Heads * prompt * Dim), yp = on.Random(Heads * prompt * Dim), lse = on.Random(Heads * prompt);
            long causal = (long)Heads * prompt * (prompt + 1) / 2 * Dim;
            BenchOperation(b, string.Create(CultureInfo.InvariantCulture, $"tiled attention with the log-sum-exp, {Heads} heads x {prompt} rows (causal), head {Dim}"), causal,
                HipBackend.Cost("attention_decode_f32"), () => hip.AttentionTiled(qp, keys, values, start, yp, lse, Heads, prompt, prompt, capacity, Dim, Scale));
            Storage keysS = on.Random(KvHeads * prompt * Dim), valuesS = on.Random(KvHeads * prompt * Dim);
            Storage starts = on.Of(new float[prompt]), ends = on.Of([.. Enumerable.Range(1, prompt).Select(i => (float)i)]);
            BenchOperation(b, string.Create(CultureInfo.InvariantCulture, $"attention over key ranges, {Heads} heads over {KvHeads}, {prompt} rows (causal), head {Dim}"), causal,
                HipBackend.Cost("attention_spans_f32"), () => hip.AttentionSpans(qp, keysS, valuesS, starts, ends, yp, lse, Heads, KvHeads, Heads, prompt, prompt, Dim, Scale));

            // The int8 cache's composed path: its scores, then its weighted values.
            Storage scores = on.Random(Heads * capacity), weights = on.Random(Heads * capacity, 0f, 1f / capacity);
            BenchOperation(b, $"int8 cache scores, {shape}", decoding, HipBackend.Cost("attn_scores_int8"),
                () => hip.AttentionScoresInt8(q, keys8, keyScales, scores, Heads, 1, capacity, Dim));
            BenchOperation(b, $"int8 cache weighted values, {shape}", decoding, HipBackend.Cost("attn_context_int8"),
                () => hip.AttentionContextInt8(weights, values8, valueScales, y, Heads, 1, capacity, Dim));
        }

        long host = Idrak.Abstraction.Operations.Kernels.HostCalls(hip) - before;
        b.Row("  (host fallbacks during the section)", host == 0 ? "none: every operation ran on its kernel" : $"{host} (SHOULD BE NONE)");
    }

    private static void BenchHipShapes(HipBench b)
    {
        if (!BenchKernelSet(b))
        {
            return;
        }

        var hip = b.Hip;
        long before = Idrak.Abstraction.Operations.Kernels.HostCalls(hip);
        const int Heads = 32, Dim = 128, Steps = 16, Width = 1024, Count = 512;
        int capacity = (int)Math.Clamp(b.LargeElements / (4L * Heads * Dim), 64, 8192);
        int vocabulary = (int)Math.Clamp(b.LargeElements / (4L * Width), Count, 1 << 18);
        int cols = 4096, rows = Math.Max(1, b.LargeElements / 4 / cols);
        using (var on = new BenchStorages(hip, new Random(8)))
        {
            Storage source = on.Random(Heads * Steps * Dim), at = on.Of([capacity - Steps]);
            Storage cache = on.Random(Heads * capacity * Dim), cache16 = on.Random(Heads * capacity * (Dim / 2)), cache8 = on.Random(Heads * capacity * (Dim / 4));
            Storage scales = on.Random(Heads * capacity), mask = on.Random(Steps * capacity);
            long written = (long)Heads * Steps * Dim;
            string steps = string.Create(CultureInfo.InvariantCulture, $"{Heads} heads x {Steps} steps, head {Dim}");
            BenchOperation(b, $"cache write, float32, {steps}", written, HipBackend.Cost("kv_write_f32"), () => hip.KeyValueWrite(source, cache, at, Heads, Steps, capacity, Dim));
            BenchOperation(b, $"cache write, bfloat16, {steps}", written, HipBackend.Cost("kv_write_bf16"),
                () => hip.KeyValueWriteBFloat16(source, cache16, at, Heads, Steps, capacity, Dim));
            BenchOperation(b, $"cache write, int8, {steps}", written, HipBackend.Cost("kv_write_int8"),
                () => hip.KeyValueWriteInt8(source, cache8, scales, at, Heads, Steps, capacity, Dim));
            BenchOperation(b, string.Create(CultureInfo.InvariantCulture, $"decoder mask {Steps}x{capacity}"), (long)Steps * capacity, HipBackend.Cost("decoder_mask_f32"),
                () => hip.DecoderMask(at, mask, Steps, capacity));

            var random = new Random(9);
            Storage ids = on.Of([.. Enumerable.Range(0, Count).Select(_ => (float)random.Next(vocabulary))]);
            Storage table = on.Random(vocabulary * Width), table16 = on.Random(vocabulary * (Width / 2)), gathered = on.Random(Count * Width);
            string gather = string.Create(CultureInfo.InvariantCulture, $"{Count} of {vocabulary} rows of {Width}");
            BenchOperation(b, $"gather, float32, {gather}", (long)Count * Width, HipBackend.Cost("gather_f32"), () => hip.Gather(table, ids, gathered, Count, Width, vocabulary));
            BenchOperation(b, $"gather, bfloat16, {gather}", (long)Count * Width, HipBackend.Cost("gather_bf16_f32"),
                () => hip.GatherBFloat16(table16, ids, gathered, Count, Width, vocabulary));
            int columnsDim = Math.Max(1, vocabulary * (Width / 2) / ((vocabulary + 1) / 2));
            BenchOperation(b, string.Create(CultureInfo.InvariantCulture, $"gather of columns, bfloat16, {Count} of {vocabulary} columns of {columnsDim}"), (long)Count * columnsDim,
                HipBackend.Cost("gather_bf16_cols_f32"), () => hip.GatherBFloat16Columns(table16, ids, gathered, Count, Math.Min(columnsDim, Width), vocabulary));

            Storage x = on.Random(rows * cols), z = on.Random(rows * cols);
            long n = (long)rows * cols;
            string matrix = string.Create(CultureInfo.InvariantCulture, $"{rows}x{cols}");
            BenchOperation(b, $"permute (a transpose) {matrix}", n, HipBackend.Cost("permute_f32"), () => hip.Permute(x, z, [cols, rows], [1, cols], accumulate: false));
            BenchOperation(b, $"strided copy, adding, {matrix}", n, HipBackend.Cost("copy2d_f32", variant: true),
                () => hip.Copy2D(x, 0, cols, z, 0, cols, rows, cols, accumulate: true));
        }

        long host = Idrak.Abstraction.Operations.Kernels.HostCalls(hip) - before;
        b.Row("  (host fallbacks during the section)", host == 0 ? "none: every operation ran on its kernel" : $"{host} (SHOULD BE NONE)");
    }
}
