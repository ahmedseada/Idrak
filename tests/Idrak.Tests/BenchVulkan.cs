// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak;
using Idrak.Backends;
using Idrak.Backends.Vulkan;
using Idrak.Generation;
using Idrak.Layers;

// dotnet run -c Release --project tests/Idrak.Tests -- --bench-vulkan [dispatch|copies|matmul|gemv|attention|sampling|window|decoder …]

// Every Vulkan device found (listed or not, so software drivers too), or the ones IDRAK_DEVICES names (vulkan:0, …):
// dispatch overhead, copies, element-wise bandwidth, matrix products, decoding-sized packed products, decoding attention
// and whole decoders generating text (all of them, or the sections named).
internal static partial class Tests
{
    internal static int BenchVulkan(string[] sections)
    {
        List<Device> devices = Environment.GetEnvironmentVariable("IDRAK_DEVICES") is { Length: > 0 } chosen
            ? [.. chosen.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(Device.Parse).Where(d => d.Type == DeviceType.Vulkan)]
            : [.. Enumerable.Range(0, VulkanBackend.DeviceCount).Select(i => Device.Get("vulkan", i))];
        if (devices.Count == 0)
        {
            Console.WriteLine($"no Vulkan device ({(VulkanBackend.DeviceCount == 0 ? VulkanBackend.UnavailableReason : "IDRAK_DEVICES names none")})");
            return 1;
        }

        foreach (var device in devices)
        {
            BenchVulkanDevice(device, sections.Length == 0 ? null : [.. sections]);
        }

        return 0;
    }

    // An empty kernel (one binding it never touches): what a dispatch costs with no work in it.
    private static readonly Lazy<VulkanKernel> EmptyKernel = new(() =>
    {
        var k = new KernelBuilder("empty", 64);
        k.Buffer("unused");
        var built = k.Build();
        return new VulkanKernel(built.Words, built.Bindings, built.PushBytes, built.Name);
    });

    private static void BenchVulkanDevice(Device device, HashSet<string>? sections)
    {
        bool Run(string section) => sections is null || sections.Contains(section);
        var backend = (VulkanBackend)device.Backend;
        Console.WriteLine($"== {device}: {backend.Describe()}");
        Console.WriteLine($"   kernels: {backend.Limits}; choices measured on first use{(VulkanBackend.Autotune ? "" : " off (IDRAK_AUTOTUNE=0)")}, stored in {VulkanBackend.TuningCacheFile ?? "memory only"}");
        var random = new Random(5);
        float[] Values(long n)
        {
            var values = new float[n];
            for (long i = 0; i < n; i++)
            {
                values[i] = random.NextSingle() - 0.5f;
            }

            return values;
        }

        static string Row(string name, string value) => $"  {name,-52} {value}";

        // The kernels one run of `run` dispatched (its measured choice already settled): "cooperative matrices
        // (coop_f32_…)" when a cooperative-matrix kernel ran, else the float32 kernels' names.
        string MatrixPath(Action run)
        {
            var kernels = new System.Collections.Concurrent.ConcurrentDictionary<string, long>();
            backend.DispatchesByKernel = kernels;
            try
            {
                run();
                device.Synchronize();
            }
            finally
            {
                backend.DispatchesByKernel = null;
            }

            string names = string.Join(", ", kernels.Keys.Order(StringComparer.Ordinal));
            return kernels.Keys.Any(k => k.Contains("_round_", StringComparison.Ordinal)) ? $"cooperative matrices, reduced precision ({names})"
                : kernels.Keys.Any(k => k.StartsWith("coop_", StringComparison.Ordinal)) ? $"cooperative matrices ({names})" : $"no matrix units ({names})";
        }

        // Average time of `run` in µs over `repeats` calls (after one warm-up), the device drained before and after.
        double Micros(Action run, int repeats)
        {
            run();
            device.Synchronize();
            var watch = Stopwatch.StartNew();
            for (int i = 0; i < repeats; i++)
            {
                run();
            }

            device.Synchronize();
            return watch.Elapsed.TotalMilliseconds * 1000 / repeats;
        }

        // ---- dispatch overhead
        if (Run("dispatch"))
        {
            var one = backend.Allocate(64, zeroed: true);
            var many = Enumerable.Range(0, 8).Select(_ => backend.Allocate(64, zeroed: true)).ToArray();
            var push = new byte[8];
            BitConverter.TryWriteBytes(push, 1);
            BitConverter.TryWriteBytes(push.AsSpan(4), 2f);
            var built = VulkanKernels.Get("fill");
            var fill = new VulkanKernel(built.Words, built.Bindings, built.PushBytes, built.Name, built.Writes);
            long submissions = backend.Submissions, barriers = backend.Barriers, dispatches = backend.Dispatches;
            const int Batch = 2000;
            Console.WriteLine(Row("descriptors", backend.PushDescriptors ? "pushed (VK_KHR_push_descriptor)" : "descriptor sets"));
            Console.WriteLine(Row("empty kernel, batched", $"{Micros(() => backend.Dispatch(EmptyKernel.Value, 1, 1, 1, [one], []), Batch):F2} µs per dispatch"));
            Console.WriteLine(Row("tiny kernel (fill 1), same buffer: a barrier each", $"{Micros(() => backend.Dispatch(fill, 1, 1, 1, [one], push), Batch):F2} µs per dispatch"));
            int next = 0;
            Console.WriteLine(Row("tiny kernel, 8 buffers in turn: a barrier every 8", $"{Micros(() => backend.Dispatch(fill, 1, 1, 1, [many[next++ & 7]], push), Batch):F2} µs per dispatch"));
            Console.WriteLine(Row("tiny kernel then wait (round trip)", $"{Micros(() => { backend.Dispatch(fill, 1, 1, 1, [one], push); backend.Synchronize(); }, 200):F1} µs"));
            Console.WriteLine(Row("host fallback of a tiny operation (AxpyAt via HostCall)", $"{Micros(() => FallbackAxpyAt(backend, one), 200):F1} µs"));
            Console.WriteLine(Row("  (submissions, barriers per dispatch above)",
                $"{backend.Submissions - submissions}, {(backend.Barriers - barriers) / (double)(backend.Dispatches - dispatches):F2}"));

            // The host's share: recording 200 dispatches (fewer than a batch holds, so nothing is submitted or waited for).
            device.Synchronize();
            long allocated = GC.GetAllocatedBytesForCurrentThread();
            var watch = Stopwatch.StartNew();
            for (int i = 0; i < 200; i++)
            {
                backend.Dispatch(fill, 1, 1, 1, [many[i & 7]], push);
            }

            double recording = watch.Elapsed.TotalMilliseconds * 1000 / 200;
            allocated = GC.GetAllocatedBytesForCurrentThread() - allocated;
            device.Synchronize();
            Console.WriteLine(Row("recording a dispatch (host side only)", $"{recording:F2} µs, {allocated / 200.0:F0} bytes allocated"));
            one.Release();
            foreach (var s in many)
            {
                s.Release();
            }
        }

        // ---- copies and bandwidth
        if (Run("copies"))
        {
            const int N = 16 << 20;                                          // 16M floats, 64 MB
            var host = Values(N);
            var a = backend.Allocate(N, zeroed: false);
            var b = backend.Allocate(N, zeroed: false);
            var c = backend.Allocate(N, zeroed: false);
            double up = Micros(() => backend.Upload(host, a), 5);
            double down = Micros(() => backend.Download(a, host), 5);
            Console.WriteLine(Row("upload 64 MB", $"{N * 4.0 / (up * 1e3):F2} GB/s"));
            Console.WriteLine(Row("download 64 MB", $"{N * 4.0 / (down * 1e3):F2} GB/s"));
            backend.Upload(host, b);
            double add = Micros(() => backend.Binary(BinaryOp.Add, a, b, c, N), 10);
            Console.WriteLine(Row("element-wise c = a + b, 16M floats", $"{3.0 * N * 4 / (add * 1e3):F1} GB/s ({add / 1000:F2} ms)"));
            double axpy = Micros(() => backend.Axpy(a, c, N, 0.5f), 10);
            Console.WriteLine(Row("element-wise y += alpha x, 16M floats", $"{3.0 * N * 4 / (axpy * 1e3):F1} GB/s ({axpy / 1000:F2} ms)"));
            a.Release();
            b.Release();
            c.Release();
        }

        // ---- matrix products
        double? gflops1024 = null;
        foreach (int size in Run("matmul") ? new[] { 1024, 4096 } : [])
        {
            long elements = (long)size * size;
            if (size == 4096)
            {
                // Skip what would take over two minutes at the 1024³ rate, or does not fit.
                double seconds = 2.0 * size * size * size * 3 / ((gflops1024 ?? 1) * 1e9);
                if (seconds > 120 || elements * 4 * 3 > backend.MaxStorageBytes * 3)
                {
                    Console.WriteLine(Row($"matrix product {size}³", $"skipped (about {seconds:F0} s at the 1024³ rate)"));
                    continue;
                }
            }

            var a = backend.Allocate((int)elements, zeroed: false);
            var b = backend.Allocate((int)elements, zeroed: false);
            var c = backend.Allocate((int)elements, zeroed: false);
            var values = Values(elements);
            backend.Upload(values, a);
            backend.Upload(values, b);
            int repeats = size == 1024 ? 5 : 2;
            void Product() => backend.BatchedMatMul(a, b, c, 1, size, size, size, false, false, 0f);
            double us = Micros(Product, repeats);
            double gflops = 2.0 * size * size * size / (us * 1e3);
            gflops1024 ??= gflops;
            Console.WriteLine(Row($"matrix product {size}³ (float32)", $"{gflops:F1} GFLOP/s ({us / 1000:F1} ms), {MatrixPath(Product)}"));

            // Each kernel forced once (the device's width), what the measured choice picks among: tiled, register-blocked,
            // cooperative matrices at float32 accuracy, and with MixedPrecision bfloat16 the reduced-precision kernel.
            string Candidates(bool mixed)
            {
                var parts = new List<string>();
                foreach (var (kernel, name) in new[] { (1, "tiled"), (2, "blocked"), (3, "matrix units, float32-accurate"), (4, "matrix units, reduced precision") })
                {
                    if (kernel == 4 && !mixed)
                    {
                        continue;
                    }

                    VulkanBackend.MatMulKernel = kernel;
                    try
                    {
                        string path = MatrixPath(Product);
                        bool ran = kernel switch { 3 => path.Contains("coop_") && !path.Contains("_round_"), 4 => path.Contains("_round_"), _ => !path.Contains("coop_") };
                        parts.Add(ran ? $"{name} {Micros(Product, repeats) / 1000:F1} ms" : $"{name} n/a");
                    }
                    finally
                    {
                        VulkanBackend.MatMulKernel = null;
                    }
                }

                return string.Join(", ", parts);
            }

            bool timeEach = us < 1e6;                                           // each kernel timed when a product takes under a second
            if (timeEach)
            {
                Console.WriteLine(Row("  kernels (float32)", Candidates(mixed: false)));
            }

            using (MixedPrecision.BFloat16())
            {
                double mixedUs = Micros(Product, repeats);
                Console.WriteLine(Row($"matrix product {size}³ (MixedPrecision bfloat16)",
                    $"{2.0 * size * size * size / (mixedUs * 1e3):F1} GFLOP/s ({mixedUs / 1000:F1} ms), {MatrixPath(Product)}"));
                if (timeEach && backend.MatrixShape is not null)
                {
                    Console.WriteLine(Row("  kernels (bfloat16)", Candidates(mixed: true)));
                }
            }
            a.Release();
            b.Release();
            c.Release();
        }

        // ---- decoding-sized packed products (one row)
        if (Run("gemv"))
        {
            using var noGrad = Autograd.NoGrad();
            foreach (var (k, n) in new[] { (1024, 3072), (3072, 1024), (1024, 151_936) })
            {
                var weights = Values((long)k * n);
                using var x = Tensor.From(Values(k), [1, k], device);
                using var int8 = Int8Weight.Quantize(weights, k, n, device);
                using var int4 = Int4Weight.Quantize(weights, k, n, device);
                using var bf16 = BFloat16Weight.FromValues(weights, k, n, device);
                foreach (var (format, bytes, run) in new (string, double, Action)[]
                {
                    ("int8", (double)k * n, () => x.MatMulInt8(int8).Dispose()),
                    ("int4", k * n / 2.0, () => x.MatMulInt4(int4).Dispose()),
                    ("bfloat16", k * n * 2.0, () => x.MatMulBFloat16(bf16).Dispose()),
                })
                {
                    long fallbacks = backend.HostCalls;
                    double us = Micros(run, n > 100_000 ? 5 : 20);
                    string note = backend.HostCalls > fallbacks ? " (host fallback: the weights exceed one binding)" : "";
                    Console.WriteLine(Row($"{format} product 1 x {k} -> {n}", $"{us,9:F1} µs ({bytes / (us * 1e3):F1} GB/s of weights){note}"));
                }
            }
        }

        // ---- decoding attention: one new row per query head (8 key/value heads × 2 query heads, head size 128)
        if (Run("attention"))
        {
            using var noGrad = Autograd.NoGrad();
            const int Heads = 8, Dim = 128, Capacity = 4096;
            using var q = Tensor.From(Values(Heads * 2 * Dim), [Heads, 2, Dim], device);
            foreach (var format in new[] { KeyValueFormat.Float32, KeyValueFormat.Int8, KeyValueFormat.BFloat16 })
            {
                using var cache = new KeyValueCache(Heads, Capacity, Dim, device, format);
                var line = new System.Text.StringBuilder();
                foreach (int length in new[] { 200, 1000, 4000 })
                {
                    using var position = Tensor.From([length - 1f], [1], device);
                    Action run = format switch
                    {
                        KeyValueFormat.Float32 => () => TensorLayerPaths.AttentionDecode(q, cache, position, 1, 0.088f).Dispose(),
                        KeyValueFormat.Int8 => () => TensorLayerPaths.AttentionInt8(q, cache, position, 1, 0.088f, tiled: false).Dispose(),
                        _ => () => TensorLayerPaths.AttentionBFloat16(q, cache, position, 1, 0.088f, tiled: false).Dispose(),
                    };
                    line.Append($"{length}: {Micros(run, 10),8:F1} µs  ");
                }

                Console.WriteLine(Row($"attention decode, {format} cache", line.ToString()));
            }
        }

        // ---- sampling one token (penalties, the draw, the history), as a decoding step does
        if (Run("sampling"))
        {
            foreach (int vocabulary in new[] { 40, 151_936 })
            {
                using var logits = Tensor.From([.. Values(vocabulary).Select(v => v * 16)], [1, vocabulary], device);
                foreach (var (name, topK, topP, penalty) in new[] { ("plain", 0, 1f, 1f), ("top-k 40", 40, 1f, 1f), ("top-k 20, top-p 0.95, penalties", 20, 0.95f, 1.1f), ("top-p 0.95", 0, 0.95f, 1f) })
                {
                    using var sampler = new TokenSampler(device, 1, vocabulary, 1000) { Temperature = 0.7f, TopK = topK, TopP = topP, RepeatPenalty = penalty, Seed = 1 };
                    sampler.SetHistory([1, 2, 3, 2]);
                    long fallbacks = backend.HostCalls;
                    double us = Micros(() => sampler.Sample(logits), 50);
                    string note = backend.HostCalls > fallbacks ? " (host fallback)" : "";
                    Console.WriteLine(Row($"sampling {vocabulary} tokens, {name}", $"{us,9:F1} µs{note}"));
                    sampler.Reset();
                }
            }
        }

        // ---- sliding windows and soft-caps (BenchWindowDevice), with this device's dispatch counts
        if (Run("window"))
        {
            BenchWindowDevice(device, () => (backend.Dispatches, backend.HostCalls));
        }

        // ---- whole decoders generating text
        if (!Run("decoder"))
        {
            return;
        }

        var tokenizer = new CharTokenizer("abcdefghijklmnopqrstuvwxyz .,<>|/_{}\":\n");
        var small = new DecoderSpec
        {
            Vocabulary = tokenizer.VocabularySize, Dim = 16, Layers = 2, Heads = 4, KvHeads = 2, HeadDim = 6, FfDim = 24, MaxPositions = 256,
            Rope = new RopeSettings(500f), NormEpsilon = 1e-5f,
        };
        var medium = new DecoderSpec
        {
            Vocabulary = tokenizer.VocabularySize, Dim = 1024, Layers = 8, Heads = 16, KvHeads = 8, HeadDim = 64, FfDim = 3072, MaxPositions = 256,
            Rope = new RopeSettings(1_000_000f), NormEpsilon = 1e-6f, QkNorm = true,
        };
        var sampling = new GenerationOptions { Seed = 3, NumPredict = 64, Temperature = 0.8f, TopK = 40, TopP = 0.95f, RepeatPenalty = 1.1f };
        foreach (var (name, spec, options, packed, tokens) in new (string, DecoderSpec, DecoderBuildOptions, string, int)[]
        {
            ("small decoder (dim 16, 2 layers), float32", small, new DecoderBuildOptions { Device = device, Seed = 1 }, "", 64),
            ("medium decoder (dim 1024, 8 layers), int8", medium, new DecoderBuildOptions { Device = device, Seed = 1, Int8 = true }, "int8", 32),
        })
        {
            using var model = spec.Build(null, options);
            var generator = new TextGenerator(model, tokenizer, 256) { KeepCache = false };
            var run = sampling with { NumPredict = tokens };
            generator.Generate("hello", run with { NumPredict = 4 });                // warm-up: pipelines built
            long fallbacks = backend.HostCalls, dispatches = backend.Dispatches;
            var stats = generator.Generate("hello", run).Stats;
            double perToken = 1.0 / Math.Max(1, stats.GeneratedTokens);
            Console.WriteLine(Row($"{name}", $"{stats.TokensPerSecond:F1} tokens/s ({(backend.Dispatches - dispatches) * perToken:F0} dispatches, "
                + $"{(backend.HostCalls - fallbacks) * perToken:F1} host fallbacks per token)"));

            // The same generation again (choices measured by the first one are known now), and without recorded graphs
            // (every step recorded by the host), so a slow replay or a slow first use shows apart from the kernels.
            var again = generator.Generate("hello", run).Stats;
            long direct = backend.Dispatches;
            var plain = generator.Generate("hello", run with { UseGraph = false }).Stats;
            Console.WriteLine(Row("    again; without graphs", $"{again.TokensPerSecond:F1} tokens/s; {plain.TokensPerSecond:F1} tokens/s "
                + $"({(backend.Dispatches - direct) / (double)Math.Max(1, plain.GeneratedTokens):F0} dispatches per token)"));

            // One more run counting by kernel and by fallback (prompt included, so the counts are per generation).
            var kernels = new System.Collections.Concurrent.ConcurrentDictionary<string, long>();
            var operations = new System.Collections.Concurrent.ConcurrentDictionary<string, long>();
            (backend.DispatchesByKernel, backend.HostCallsByOperation) = (kernels, operations);
            try
            {
                generator.Generate("hello", run);
            }
            finally
            {
                (backend.DispatchesByKernel, backend.HostCallsByOperation) = (null, null);
            }

            Console.WriteLine(Row("  kernels per token", string.Join(", ", kernels.OrderByDescending(p => p.Value).Select(p => $"{p.Key} {p.Value * perToken:F1}"))));
            if (!operations.IsEmpty)
            {
                Console.WriteLine(Row("  host fallbacks per token", string.Join(", ", operations.OrderByDescending(p => p.Value).Select(p => $"{p.Key} {p.Value * perToken:F1}"))));
            }
        }

        Console.WriteLine();
    }

    // AxpyAt through the base class's host fallback (download, CPU, upload), for comparison with a dispatch.
    private static void FallbackAxpyAt(VulkanBackend backend, Storage storage)
    {
        using var call = new HostCall(backend);
        Idrak.Abstraction.Devices.Cpu.CpuBackend.Instance.AxpyAt(call[storage], call[storage], 1, 0.5f);
    }
}
