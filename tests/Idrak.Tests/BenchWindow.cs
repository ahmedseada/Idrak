// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models.Abstractions;

// dotnet run -c Release --project tests/Idrak.Tests -- --bench-window
// Sliding windows and soft-caps on every device IDRAK_DEVICES names (default: every GPU found, else the CPU): decoding
// attention from each row's window start against no window, and a windowed, soft-capped decoder generating through the
// attention kernels against the composed path such layers took before (IDRAK_WINDOW_KERNELS=0: scores over every cached
// slot, masked). Also --bench-vulkan window, which adds the dispatch counts.
internal static partial class Tests
{
    internal static int BenchWindow()
    {
        List<Device> devices = Environment.GetEnvironmentVariable("IDRAK_DEVICES") is { Length: > 0 } chosen
            ? [.. chosen.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(Device.Parse)]
            : [.. Device.Available.Where(d => d.Type != DeviceType.Cpu).DefaultIfEmpty(Device.Cpu)];
        foreach (var device in devices)
        {
            Console.WriteLine($"== {device}");
            BenchWindowDevice(device, null);
            Console.WriteLine();
        }

        return 0;
    }

    // counters: the device's (dispatches, host fallbacks) so far, when it counts them.
    private static void BenchWindowDevice(Device device, Func<(long Dispatches, long HostCalls)>? counters)
    {
        static string Row(string name, string value) => $"  {name,-60} {value}";
        var random = new Random(5);

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

        using (Autograd.NoGrad())
        {
            const int Heads = 8, Dim = 128, Capacity = 4096;
            using var q = Tensor.From([.. Enumerable.Range(0, Heads * 2 * Dim).Select(_ => random.NextSingle() - 0.5f)], [Heads, 2, Dim], device);
            using var cache = new KeyValueCache(Heads, Capacity, Dim, device, KeyValueFormat.Float32);
            foreach (var (label, variant) in new[] { ("no window", default(AttentionVariant)), ("window 512", new AttentionVariant(512, 0f)), ("window 512, soft-cap 50", new AttentionVariant(512, 50f)) })
            {
                var line = new System.Text.StringBuilder();
                foreach (int length in new[] { 200, 1000, 4000 })
                {
                    using var position = Tensor.From([length - 1f], [1], device);
                    line.Append($"{length}: {Micros(() => Tensor.AttentionDecode(q, cache, position, 1, 0.088f, variant).Dispose(), 10),8:F1} µs  ");
                }

                Console.WriteLine(Row($"attention decode, float32 cache, {label}", line.ToString()));
            }
        }

        var letters = new CharTokenizer("abcdefghijklmnopqrstuvwxyz .,<>|/_{}\":\n");
        var windowed = new DecoderSpec
        {
            Vocabulary = letters.VocabularySize, Dim = 256, Layers = 4, Heads = 4, KvHeads = 2, HeadDim = 64, FfDim = 512, MaxPositions = 1024,
            Rope = new RopeSettings(10000f), NormEpsilon = 1e-6f, SlidingWindow = 128, SlidingWindowLayers = [true, false, true, false],
            AttentionSoftcap = 50f, LogitSoftcap = 30f,
        };
        string prompt = string.Concat(Enumerable.Repeat("the quick brown fox jumps over the lazy dog. ", 14));
        var options = new GenerationOptions { Seed = 3, NumPredict = 32, Temperature = 0.8f, TopK = 40 };
        using var model = windowed.Build(null, new DecoderBuildOptions { Device = device, Seed = 1 });
        var generator = new TextGenerator(model, letters, 1024) { KeepCache = false };
        bool saved = CausalSelfAttention.WindowKernels;
        try
        {
            foreach (bool kernels in new[] { false, true })
            {
                CausalSelfAttention.WindowKernels = kernels;
                generator.Generate(prompt, options with { NumPredict = 4 });       // warm-up: kernels built, choices measured
                var before = counters?.Invoke();
                var stats = generator.Generate(prompt, options).Stats;
                double perToken = 1.0 / Math.Max(1, stats.GeneratedTokens);
                string counted = counters?.Invoke() is { } after && before is { } start
                    ? $" ({(after.Dispatches - start.Dispatches) * perToken:F0} dispatches, {(after.HostCalls - start.HostCalls) * perToken:F1} host fallbacks per token)"
                    : "";
                Console.WriteLine(Row($"windowed decoder (dim 256, 4 layers, window 128, caps), {(kernels ? "kernels" : "composed")}",
                    $"{stats.TokensPerSecond:F1} tokens/s, prompt of {stats.PromptTokens} in {stats.PromptDuration.TotalMilliseconds:F0} ms{counted}"));
            }
        }
        finally
        {
            CausalSelfAttention.WindowKernels = saved;
        }
    }
}
