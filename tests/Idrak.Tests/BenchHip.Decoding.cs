// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Gpu.Hip;

// --bench-hip decoding (plan 35, G5a): HIP's decoding operations on their GCN kernels, each timed cold (its first call)
// apart from warm (BenchOperation), with GB/s and GFLOP/s from the tables of GcnKernels.Decoding.cs, .Products.cs and
// .Pointwise.cs (HipBackend.Cost, HipBackend.ProductCost), ending with the host fallbacks the section took (none expected).
// Sizes follow the large buffer (from the memory the device reports), never a card's: a layer of k ≤ 4,096 inputs and
// 4,096 outputs for the fused products (1 and 8 rows, int8, int4 and bfloat16), a cache of up to 4,096 positions of 32
// heads of 128 for the rows' attention, packed sequences of up to 512 positions, a vocabulary within a quarter of it
// (the sampler's three paths over it: greedy rows, top-k 40 through the candidates, top-p over the whole row), float32
// products of one input within a third of it.
internal abstract partial class DesktopTests
{
    private static void BenchHipDecoding(HipBench b)
    {
        if (!BenchKernelSet(b))
        {
            return;
        }

        var hip = b.Hip;
        long before = Idrak.Abstraction.Operations.Kernels.HostCalls(hip);
        var random = new Random(9);
        using (var on = new BenchStorages(hip, random))
        {
            // Packed weights of random bits: small integers for int8 and int4, halves near 1 for bfloat16.
            Storage Packed(long words, bool halves)
            {
                var values = new float[Math.Max(1, words)];
                var bits = System.Runtime.InteropServices.MemoryMarshal.Cast<float, uint>(values.AsSpan());
                for (int i = 0; i < bits.Length; i++)
                {
                    bits[i] = halves ? 0x3F803F80u ^ (uint)random.Next(0, 0x007F007F) : (uint)random.Next();
                }

                return on.Of(values);
            }

            int k = (int)Math.Clamp(b.LargeElements / 4096L, 256, 4096) / 64 * 64, n = 4096;
            Storage x = on.Random(8 * k), up = on.Random(8 * k), y = on.Random(8 * n), y2 = on.Random(8 * n), y3 = on.Random(8 * n), hidden = on.Random(8 * n);
            Storage residual = on.Random(8 * n), sum = on.Random(8 * n), normalized = on.Random(8 * n), gain = on.Random(n), bias = on.Random(n);
            foreach (var format in new[] { PackedFormat.Int8, PackedFormat.Int4, PackedFormat.BFloat16 })
            {
                var weights = format switch { PackedFormat.Int8 => GcnKernels.PackedWeights.Int8, PackedFormat.Int4 => GcnKernels.PackedWeights.Int4, _ => GcnKernels.PackedWeights.BFloat16 };
                int perWord = GcnKernels.ColumnsPerWord(weights);
                long words = (n + perWord - 1) / perWord;
                Storage w1 = Packed(k * words, format == PackedFormat.BFloat16), w2 = Packed(k * words, format == PackedFormat.BFloat16);
                Storage? scales = format == PackedFormat.BFloat16 ? null : on.Random(format == PackedFormat.Int4 ? (k + 31) / 32 * 8 * (int)words : n, 0.001f, 0.01f);
                string name = format switch { PackedFormat.Int8 => "int8", PackedFormat.Int4 => "int4", _ => "bf16" };
                foreach (int m in new[] { 1, 8 })
                {
                    var cost = HipBackend.ProductCost($"{name}_gemv_f32", m, n, k);
                    BenchProduct(b, string.Create(CultureInfo.InvariantCulture, $"{name} gated (SiLU) {m}x{k}->{n}"), cost,
                        () => hip.PackedMatMulGated(format, 0, x, up, w1, scales, y, m, n, k));
                    BenchProduct(b, string.Create(CultureInfo.InvariantCulture, $"{name} with residual and norm {m}x{k}->{n}"), cost,
                        () => hip.PackedMatMulAddRmsNorm(format, x, w1, scales, y, m, n, k, residual, sum, gain, normalized, 1e-6f, 0f));
                    BenchProduct(b, string.Create(CultureInfo.InvariantCulture, $"{name} norm, residual and norm {m}x{k}->{n}"), cost,
                        () => hip.PackedMatMulNormAddRmsNorm(format, x, w1, scales, y, m, n, k, gain, 1e-6f, 0f, residual, sum, gain, normalized, 1e-6f, 0f));
                    var three = HipBackend.ProductCost($"{name}_gemv_f32", m, 3 * n, k);
                    BenchProduct(b, string.Create(CultureInfo.InvariantCulture, $"{name} three products of one input {m}x{k}->3x{n}"), three,
                        () => hip.PackedMatMulMany(format, x, m, k, [(w1, scales, bias, y, n), (w2, scales, null, y2, n), (w1, scales, null, y3, n)]));
                    var two = HipBackend.ProductCost($"{name}_gemv_f32", m, 2 * n, k);
                    BenchProduct(b, string.Create(CultureInfo.InvariantCulture, $"{name} gate/up pair (SiLU) {m}x{k}->2x{n}"), two,
                        () => hip.PackedMatMulGatedPair(format, 0, x, m, k, [(w1, scales, null, y, n), (w2, scales, null, y2, n)], hidden));
                }
            }

            // Attention over rows of different lengths (a float32 and a bfloat16 cache) and over packed sequences.
            const int Heads = 32, Dim = 128;
            const float Scale = 0.088f;
            int capacity = (int)Math.Clamp(b.LargeElements / (8L * Heads * Dim), 256, 4096);
            Storage q = on.Random(Heads * 4 * Dim), keys = on.Random(Heads * capacity * Dim), values = on.Random(Heads * capacity * Dim), output = on.Random(Heads * 4 * Dim);
            Storage last = on.Of([capacity - 1]), starts = on.Of([.. Enumerable.Range(0, 4).Select(i => (float)(37 * i))]);
            long decoding = (long)Heads * capacity * Dim;
            string shape = string.Create(CultureInfo.InvariantCulture, $"{Heads} heads x {capacity} positions, head {Dim}, 4 rows of different starts");
            BenchOperation(b, $"attention over rows, float32 cache, {shape}", decoding, HipBackend.Cost("attention_rows_f32"),
                () => hip.AttentionRows(q, keys, values, last, output, starts, Heads, Heads / 4, 1, 1, capacity, Dim, Scale));
            Storage keys16 = on.Random(Heads * capacity * (Dim / 2)), values16 = on.Random(Heads * capacity * (Dim / 2));
            BenchOperation(b, $"attention over rows, bfloat16 cache, {shape}", decoding, HipBackend.Cost("attention_rows_bf16"),
                () => hip.AttentionRowsBFloat16(q, keys16, values16, last, output, starts, Heads, Heads / 4, 1, 1, capacity, Dim, Scale));
            int steps = Math.Min(capacity, 512);
            Storage qs = on.Random(Heads * steps * Dim), ks = on.Random(Heads * steps * Dim), vs = on.Random(Heads * steps * Dim), ys = on.Random(Heads * steps * Dim);
            Storage lse = on.Random(Heads * steps), segmentStarts = on.Of([.. Enumerable.Range(0, steps).Select(i => (float)(i / 128 * 128))]);
            Storage segmentEnds = on.Of([.. Enumerable.Range(0, steps).Select(i => (float)Math.Min(steps, (i / 128 + 1) * 128))]);
            long causal = (long)Heads * steps * 64 * Dim;
            BenchOperation(b, string.Create(CultureInfo.InvariantCulture, $"attention over packed sequences, {Heads} heads x {steps} positions (sequences of 128), head {Dim}"),
                causal, HipBackend.Cost("attention_spans_f32"),
                () => hip.AttentionSegmented(qs, ks, vs, ys, lse, segmentStarts, segmentEnds, Heads, Heads, steps, steps, Dim, Scale));

            // The rows of decoding over a vocabulary and a model's width.
            int vocabulary = (int)Math.Clamp(b.LargeElements / 16L, 1024, 262_144), rows = 8, width = 4096;
            Storage logits = on.Random(rows * vocabulary, -8f, 8f), work = on.Random(rows * vocabulary), picks = on.Random(rows), matches = on.Random(rows);
            Storage history = on.Of([.. Enumerable.Range(0, rows * 64).Select(_ => (float)random.Next(vocabulary))]), length = on.Of([200f]);
            string vocab = string.Create(CultureInfo.InvariantCulture, $"{rows}x{vocabulary}");
            BenchOperation(b, $"arg-max {vocab}", (long)rows * vocabulary, HipBackend.Cost("argmax_f32"), () => hip.ArgMax(logits, picks, rows, vocabulary));
            BenchOperation(b, $"class match {vocab}", (long)rows * vocabulary, HipBackend.Cost("class_match_f32"), () => hip.ClassMatch(logits, work, matches, rows, vocabulary, 0.5f));
            BenchOperation(b, $"repetition penalties (a window of 64) {vocab}", (long)rows * vocabulary, HipBackend.Cost("penalize_rows_f32"),
                () => hip.PenalizeRows(logits, work, history, length, rows, vocabulary, vocabulary, 0, 64, 64, 1.1f, 0.1f, 0.05f));
            BenchOperation(b, $"history push, {rows} rows", rows, HipBackend.Cost("history_push_f32"), () => hip.HistoryPush(picks, history, length, rows, 64));
            Storage ids = on.Of([.. Enumerable.Range(0, rows).Select(i => (float)(i * 7))]), hot = on.Random(rows * vocabulary);
            BenchOperation(b, $"one-hot {vocab}", (long)rows * vocabulary, HipBackend.Cost("one_hot_f32"), () => hip.OneHot(ids, hot, rows, vocabulary));
            Storage a = on.Random(rows * width), bb = on.Random(rows * width), s = on.Random(rows * width), widthGain = on.Random(width), z = on.Random(rows * width);
            string matrix = string.Create(CultureInfo.InvariantCulture, $"{rows}x{width}");
            BenchOperation(b, $"residual add and RMS norm {matrix}", (long)rows * width, HipBackend.Cost("add_rms_norm_affine_f32"),
                () => hip.AddRmsNormAffine(a, bb, s, widthGain, z, rows, width, 1e-6f, 0f));
            int ropeHeads = Heads, ropeRows = 8 * ropeHeads, half = Dim / 2;
            Storage heads = on.Random(ropeRows * Dim), rotated = on.Random(ropeRows * Dim), cos = on.Random(capacity * half), sin = on.Random(capacity * half);
            Storage positions = on.Of([.. Enumerable.Range(0, 8).Select(i => (float)(capacity - 8 + i))]), headGain = on.Random(Dim);
            string ropeShape = string.Create(CultureInfo.InvariantCulture, $"8 steps x {ropeHeads} heads, head {Dim}");
            BenchOperation(b, $"rotary embedding, {ropeShape}", (long)ropeRows * half, HipBackend.Cost("rope_f32"),
                () => hip.Rope(heads, rotated, cos, sin, positions, ropeRows, ropeHeads, 8, Dim, half, false, 1f));
            BenchOperation(b, $"RMS norm and rotary embedding, {ropeShape}", (long)ropeRows * Dim, HipBackend.Cost("rms_norm_rope_f32"),
                () => hip.RmsNormRope(heads, headGain, cos, sin, positions, rotated, ropeRows, Dim, 1e-6f, 0f, ropeHeads, 8, half, false));

            // G5a's rest: queries and keys normalized and rotated in one launch, the attention heads into a bfloat16 cache.
            const int KvHeads = 8;
            int keyRows = 8 * KvHeads;
            Storage keyHeads = on.Random(keyRows * Dim), keyRotated = on.Random(keyRows * Dim), values8 = on.Random(keyRows * Dim);
            BenchOperation(b, $"RMS norm and rotary embedding of queries and keys, 8 steps x {ropeHeads} + {KvHeads} heads, head {Dim}", (long)(ropeRows + keyRows) * Dim,
                HipBackend.Cost("rms_norm_rope2_f32"),
                () => hip.RmsNormRopePair(heads, headGain, rotated, ropeRows, 1e-6f, 0f, ropeHeads, keyHeads, headGain, keyRotated, keyRows, 1e-6f, 0f, KvHeads, cos, sin, positions,
                    Dim, 8, half, false));
            Storage cacheKeys = on.Random(KvHeads * capacity * Dim / 2), cacheValues = on.Random(KvHeads * capacity * Dim / 2), cacheStart = on.Of([capacity - 8]);
            BenchOperation(b, $"queries, keys and values into the attention layouts and a bfloat16 cache, 8 steps x {ropeHeads} + 2 x {KvHeads} heads, head {Dim}",
                (long)(ropeRows + 2 * keyRows) * Dim, HipBackend.Cost("norm_rope_heads_f32"),
                () => hip.NormRopeHeads(heads, keyHeads, values8, 1, 8, ropeHeads, KvHeads, Dim, headGain, 1e-6f, 0f, headGain, 1e-6f, 0f, cos, sin, positions, half, false,
                    rotated, cacheKeys, cacheValues, cacheStart, capacity, Dim, true));

            // Float32 products of one input (MatMulMany: gemv_multi_f32), 1 and 8 rows.
            int floatColumns = (int)Math.Clamp(b.LargeElements / (3L * k), 256, 4096);
            Storage f1 = on.Random(k * floatColumns), f2 = on.Random(k * floatColumns), f3 = on.Random(k * floatColumns);
            foreach (int m in new[] { 1, 8 })
            {
                BenchProduct(b, string.Create(CultureInfo.InvariantCulture, $"float32 three products of one input {m}x{k}->3x{floatColumns}"),
                    HipBackend.ProductCost("gemv_multi_f32", m, 3 * floatColumns, k),
                    () => hip.MatMulMany(x, m, k, [(f1, bias, y, floatColumns), (f2, null, y2, floatColumns), (f3, null, y3, floatColumns)]));
            }

            // The sampler over the vocabulary: greedy rows, top-k 40 through the candidates, top-p 0.9 over the whole row.
            Storage sampled = on.Random(rows), samplerStats = on.Random(rows * 13), samplerStep = on.Of([0f]);
            foreach (var (label, topK, topP, kernel) in new[]
                     {
                         ("greedy (top-k 1)", 1, 1f, "sample_candidates_greedy_f32"), ("top-k 40 (the candidates)", 40, 1f, "sample_candidates_f32"),
                         ("top-p 0.9 (the whole row)", 0, 0.9f, "sample_rows_f32"),
                     })
            {
                BenchOperation(b, $"sampler, {label}, {vocab}", (long)rows * vocabulary, HipBackend.Cost(kernel),
                    () => hip.SampleRows(logits, sampled, samplerStats, samplerStep, rows, vocabulary, vocabulary, 0, 0.8f, topK, topP, 0f, 7u));
            }

            // The element-wise operations with parameters over the large buffer.
            int count = b.LargeElements / 4;
            Storage ex = on.Random(count, 0.01f, 4f), ey = on.Random(count), ez = on.Random(count), condition = on.Random(count), scalar = on.Of([0.25f]);
            string elements = string.Create(CultureInfo.InvariantCulture, $"{count} floats");
            BenchOperation(b, $"pow_f32 (x^1.5) {elements}", count, HipBackend.Cost("pow_f32"), () => hip.Pow(ex, ey, count, 1.5f));
            BenchOperation(b, $"clamp_f32 {elements}", count, HipBackend.Cost("clamp_f32"), () => hip.Clamp(ex, ey, count, 0.5f, 2f));
            BenchOperation(b, $"leaky_relu_f32 {elements}", count, HipBackend.Cost("leaky_relu_f32"), () => hip.LeakyRelu(ex, ey, count, 0.01f));
            BenchOperation(b, $"where_f32 {elements}", count, HipBackend.Cost("where_f32"), () => hip.Where(condition, ex, ey, ez, count));
            BenchOperation(b, $"inv_sqrt_f32 {elements}", count, HipBackend.Cost("inv_sqrt_f32"), () => hip.InvSqrt(ex, ey, count, 1e-6f));
            BenchOperation(b, $"bias_gelu_f32 {elements}", count, HipBackend.Cost("bias_gelu_f32"), () => hip.BiasGelu(ex, widthGain, ey, count / width * width, width));
            BenchOperation(b, $"add_scalar_f32 {elements}", count, HipBackend.Cost("add_scalar_f32"), () => hip.AddBroadcastScalar(scalar, ey, count, 0.5f));
            for (int kind = 0; kind < GcnKernels.GatedKinds.Length; kind++)
            {
                int activation = kind;
                BenchOperation(b, $"{GcnKernels.GatedKinds[kind]} {elements}", count, HipBackend.Cost(GcnKernels.GatedKinds[kind]),
                    () => hip.GatedActivation(ex, condition, ey, count, activation));
            }
        }

        long host = Idrak.Abstraction.Operations.Kernels.HostCalls(hip) - before;
        b.Row("  (host fallbacks during the section)", host == 0 ? "none: every operation ran on its kernel" : $"{host} (SHOULD BE NONE)");
    }
}
