// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Layers;

// dotnet run -c Release --project tests/Idrak.Tests -- --bench-cpu
// The CPU's decoding kernels at language-model sizes: bfloat16 products of 1-8 rows, attention of one new row over a
// float32 / int8 / bfloat16 cache of 4,000 positions (16 heads of 128), and sampling over a 151,936-token vocabulary.
internal static partial class Tests
{
    internal static int BenchCpu()
    {
        var device = Device.Cpu;
        Console.WriteLine(device.Name);
        var random = new Random(3);
        float[] Values(int n) => [.. Enumerable.Range(0, n).Select(_ => random.NextSingle() - 0.5f)];
        string Time(Action run, int repeats)
        {
            run();
            var watch = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < repeats; i++)
            {
                using var scope = new TensorScope();
                run();
            }

            return $"{watch.Elapsed.TotalMilliseconds / repeats:F2} ms";
        }

        using (Autograd.NoGrad())
        {
            const int K = 1536, N = 32_000;
            using var weights = Tensor.From(Values(K * N), [K, N], device);
            using var bf16 = BFloat16Weight.Convert(weights);
            foreach (int rows in new[] { 1, 4, 8 })
            {
                using var x = Tensor.From(Values(rows * K), [rows, K], device);
                Console.WriteLine($"{$"bfloat16 product {rows} x {K} x {N}",-44} {Time(() => x.MatMulBFloat16(bf16), 10),12}");
            }

            const int Heads = 16, Dim = 128, Filled = 4000, Capacity = 4096;
            using var q = Tensor.From(Values(Heads * Dim), [Heads, 1, Dim], device);
            using var position = Tensor.From([Filled - 1f], [1], device);
            foreach (var format in new[] { KeyValueFormat.Float32, KeyValueFormat.Int8, KeyValueFormat.BFloat16 })
            {
                using var cache = new KeyValueCache(Heads, Capacity, Dim, device, format);
                using var fill = Tensor.From(Values(Heads * Filled * Dim), [Heads, Filled, Dim], device);
                using var zero = Tensor.From([0f], [1], device);
                Action run;
                if (format == KeyValueFormat.Float32)
                {
                    Tensor.WriteKeyValues(fill, cache.Keys, zero);
                    Tensor.WriteKeyValues(fill, cache.Values, zero);
                    run = () => Tensor.AttentionDecode(q, cache, position, 1, 0.088f);
                }
                else if (format == KeyValueFormat.Int8)
                {
                    Tensor.WriteKeyValuesInt8(fill, cache.Keys, cache.KeyScales!, zero);
                    Tensor.WriteKeyValuesInt8(fill, cache.Values, cache.ValueScales!, zero);
                    run = () => Tensor.AttentionInt8(q, cache, position, 1, 0.088f, tiled: false);
                }
                else
                {
                    Tensor.WriteKeyValuesBFloat16(fill, cache.Keys, zero, Dim);
                    Tensor.WriteKeyValuesBFloat16(fill, cache.Values, zero, Dim);
                    run = () => Tensor.AttentionBFloat16(q, cache, position, 1, 0.088f, tiled: false);
                }

                Console.WriteLine($"{$"attention, {format} cache, {Filled} positions",-44} {Time(run, 20),12}");
            }

            const int Vocabulary = 151_936;
            using var logits = Tensor.From(Values(Vocabulary).Select(v => v * 16).ToArray(), [1, Vocabulary], device);
            foreach (var (name, topK, topP) in new[] { ("plain", 0, 1f), ("top-k 20", 20, 1f), ("top-k 20, top-p 0.95", 20, 0.95f), ("top-p 0.95", 0, 0.95f) })
            {
                using var sampler = new TokenSampler(device, 1, Vocabulary, 1000) { Temperature = 0.6f, TopK = topK, TopP = topP, Seed = 1 };
                Console.WriteLine($"{$"sampling {Vocabulary} tokens, {name}",-44} {Time(() => sampler.Sample(logits), 20),12}");
            }
        }

        Console.WriteLine(Idrak.Abstraction.Devices.Cpu.CpuTuning.Describe());
        using (Autograd.NoGrad())
        {
            foreach (var (m, n, k, transB) in new[]
            {
                (1, 4096, 1024, false), (4, 4096, 1024, false), (5, 4096, 1024, false), (6, 4096, 1024, false), (8, 4096, 1024, false),
                (16, 4096, 1024, false), (24, 24, 24, false), (32, 64, 64, false), (48, 48, 48, false), (64, 64, 64, false),
                (128, 128, 128, false), (512, 2048, 128, false), (512, 512, 512, false), (1024, 1024, 1024, false),
                (1, 32000, 1024, true), (16, 4096, 1024, true), (64, 4096, 1024, true),
            })
            {
                using var a = Tensor.From(Values(m * k), [m, k], device);
                using var b = transB ? Tensor.From(Values(k * n), [n, k], device) : Tensor.From(Values(k * n), [k, n], device);
                int repeats = (int)Math.Clamp(2e9 / ((double)m * n * k), 5, 20000);
                Console.WriteLine($"{$"float product {m} x {n} x {k}{(transB ? ", B transposed" : "")}",-44} {Time(() => a.MatMul(b, transposeB: transB), repeats),12}");
            }

            foreach (int n in new[] { 1 << 12, 1 << 14, 1 << 15, 1 << 16, 1 << 17, 1 << 18, 1 << 20, 1 << 22 })
            {
                using var a = Tensor.From(Values(n), [n], device);
                using var b = Tensor.From(Values(n), [n], device);
                int repeats = Math.Clamp((1 << 26) / n, 10, 20000);
                Console.WriteLine($"{$"add + sum of {n} values",-44} {Time(() => (a + b).Sum(), repeats),12}");
            }
        }

        return 0;
    }
}
