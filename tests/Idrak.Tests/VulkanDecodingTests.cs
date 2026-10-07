// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using Idrak;
using Idrak.Gpu.Vulkan;
using Idrak.Layers;

// Decoding on Vulkan: every step on the device (no host fallback, so no wait for the device between tokens), the sampler
// kernels against the CPU's, and the runtime's barriers (only between dependent commands).
internal static partial class Tests
{
    // The sampler kernels draw the CPU's tokens for the same seed and steps (greedy, temperature, top-k with and without
    // the slot stage, top-p, min-p, combined; tied scores; rows read at a stride and offset), with the CPU's statistics.
    // Sums of weights are added in another order (and exp rounds differently), so a draw can differ where the random
    // target falls within rounding of a boundary between two tokens: likely only when thousands of tokens are kept, so the
    // 10,000-token cases keep few (a top-k or min-p cut-off, or peaked scores).
    private static readonly (float Temperature, int TopK, float TopP, float MinP)[] SamplerSettings =
    [
        (1f, 1, 1f, 0f), (1f, 0, 1f, 0f), (0.7f, 40, 1f, 0f), (1.2f, 0, 0.9f, 0f), (0.8f, 0, 1f, 0.05f),
        (0.9f, 20, 0.95f, 0.02f), (1f, 64, 0.8f, 0f), (1f, 100, 1f, 0f), (0.5f, 3, 0.5f, 0.1f),
    ];

    private static void VulkanSamplerMatchesCpu(Device device) => ForEachVulkan(device, (backend, label) =>
    {
        if (Environment.GetEnvironmentVariable("IDRAK_VULKAN_KERNELS") is "0" or "false")
        {
            return;
        }

        var cpu = Idrak.Abstraction.Devices.Cpu.CpuBackend.Instance;
        var random = new Random(17);
        const int Rows = 3, Steps = 4, Positions = 2;
        foreach (int vocabulary in new[] { 50, 1000, 10_000 })              // one invocation per row, a workgroup, two stages
        {
            foreach (bool ties in new[] { false, true })
            {
                // Logits [rows, positions, vocabulary]; the sampler reads each row's last position.
                var logits = new float[Rows * Positions * vocabulary];
                bool large = vocabulary > 1000;
                for (int i = 0; i < logits.Length; i++)
                {
                    float v = large ? 4 * MathF.Sqrt(-2 * MathF.Log(1 - random.NextSingle())) * MathF.Cos(MathF.Tau * random.NextSingle())
                        : (random.NextSingle() * 2 - 1) * 6;
                    logits[i] = ties ? MathF.Round(v * 4) / 4 : v;
                }

                foreach (var (temperature, topK, topP, minP) in SamplerSettings.Where(c => !large || c.TopK > 0 || c.MinP > 0))
                {
                    string what = $"{label}: sampling {vocabulary} tokens{(ties ? " (tied scores)" : "")}, temperature {temperature}, top-k {topK}, top-p {topP}, min-p {minP}";
                    var (cpuIds, cpuStats) = SampleSteps(cpu, logits, vocabulary, temperature, topK, topP, minP);
                    long fallbacks = Idrak.Abstraction.Operations.Kernels.HostCalls(backend);
                    var (ids, stats) = SampleSteps(backend, logits, vocabulary, temperature, topK, topP, minP);
                    Check(Idrak.Abstraction.Operations.Kernels.HostCalls(backend) == fallbacks, $"{what}: took the host fallback");
                    for (int i = 0; i < stats.Length; i += 13)
                    {
                        // On a mismatch, both statistics rows and the row's largest scores, to tell a cut-off from a draw.
                        string Detail() => $" (statistics [{string.Join(", ", stats.Skip(i).Take(13))}], the CPU's [{string.Join(", ", cpuStats.Skip(i).Take(13))}]; "
                            + $"largest scores {string.Join(", ", logits.Skip((i / 13 % Rows * Positions + Positions - 1) * vocabulary).Take(vocabulary).OrderDescending().Take(4))})";
                        Check(stats[i] == cpuStats[i], $"{what}, step {i / 13 / Rows}, row {i / 13 % Rows}: token {stats[i]}, the CPU's {cpuStats[i]}" + (stats[i] == cpuStats[i] ? "" : Detail()));
                        for (int a = 0; a < 5; a++)
                        {
                            Check(stats[i + 3 + 2 * a] == cpuStats[i + 3 + 2 * a], $"{what}, step {i / 13 / Rows}, row {i / 13 % Rows}: alternative {a} is {stats[i + 3 + 2 * a]}, the CPU's {cpuStats[i + 3 + 2 * a]}");
                        }
                    }

                    Check(ids.SequenceEqual(cpuIds), $"{what}: last ids");
                    AssertClose(cpuStats, stats, 2e-4f, $"{what}: statistics");
                }
            }
        }

        // Steps of sampling on `b`: the ids of the last step and every step's statistics.
        static (float[] Ids, float[] Stats) SampleSteps(Backend b, float[] logits, int vocabulary, float temperature, int topK, float topP, float minP)
        {
            var x = b.Allocate(logits.Length, zeroed: false);
            var ids = b.Allocate(Rows, zeroed: true);
            var stats = b.Allocate(Steps * Rows * 13, zeroed: true);
            var step = b.Allocate(1, zeroed: true);
            try
            {
                b.Upload(logits, x);
                for (int s = 0; s < Steps; s++)
                {
                    b.Upload([s], step);
                    b.SampleRows(x, ids, stats, step, Rows, vocabulary, Positions * vocabulary, (Positions - 1) * vocabulary, temperature, topK, topP, minP, 1234u);
                }

                var (idValues, statValues) = (new float[Rows], new float[Steps * Rows * 13]);
                b.Download(ids, idValues);
                b.Download(stats, statValues);
                return (idValues, statValues);
            }
            finally
            {
                x.Release();
                ids.Release();
                stats.Release();
                step.Release();
            }
        }
    });

    // The sampler on a parallel device: the same input sampled again and again gives the same tokens and statistics, bit
    // for bit (a race between invocations or between the two top-k stages shows up as a run that differs). 1,000 runs on
    // a GPU, 20 on a CPU driver (which runs a workgroup's invocations in turn, so it cannot show such races anyway). Run on
    // the Vulkan device itself, as the device copies.
    private static void VulkanSamplerRepeatable(Device device)
    {
        if (device.Type != DeviceType.Vulkan || Environment.GetEnvironmentVariable("IDRAK_VULKAN_KERNELS") is "0" or "false")
        {
            return;
        }

        var backend = (VulkanBackend)device.Backend;
        string label = device.ToString();
        int runs = VulkanBackend.DeviceKind(device.Ordinal).Type == 4 ? 20 : 1000;
        const int Rows = 4;
        var random = new Random(23);
        foreach (int vocabulary in new[] { 200, 3000, 30_000 })                 // one invocation per row, a workgroup, two stages
        {
            // Tied scores in quarter steps, plus rows of long runs of equal scores (every slice of row 0 holds one value).
            var logits = new float[Rows * vocabulary];
            for (int i = 0; i < logits.Length; i++)
            {
                logits[i] = MathF.Round((random.NextSingle() * 2 - 1) * 24) / 4;
            }

            Array.Fill(logits, 0.5f, 0, vocabulary);
            Array.Fill(logits, 7f, vocabulary + vocabulary / 3, Math.Min(300, vocabulary / 3));
            var x = backend.Allocate(logits.Length, zeroed: false);
            var ids = backend.Allocate(Rows, zeroed: true);
            var stats = backend.Allocate(Rows * 13, zeroed: true);
            var step = backend.Allocate(1, zeroed: true);
            try
            {
                backend.Upload(logits, x);
                foreach (var (topK, topP, minP) in new[] { (1, 1f, 0f), (40, 0.9f, 0f), (64, 1f, 0.02f), (0, 0.95f, 0f) })
                {
                    float[]? first = null;
                    var values = new float[Rows * 13];
                    for (int run = 0; run < runs; run++)
                    {
                        backend.SampleRows(x, ids, stats, step, Rows, vocabulary, vocabulary, 0, 0.9f, topK, topP, minP, 77u);
                        backend.Download(stats, values);
                        if (first is null)
                        {
                            first = (float[])values.Clone();
                            Check(Enumerable.Range(0, Rows).All(r => values[r * 13] >= 0 && values[r * 13] < vocabulary), $"{label}: {vocabulary} tokens, top-k {topK}: tokens in range");
                            continue;
                        }

                        int at = Enumerable.Range(0, values.Length).FirstOrDefault(i => BitConverter.SingleToInt32Bits(values[i]) != BitConverter.SingleToInt32Bits(first[i]), -1);
                        Check(at < 0, $"{label}: {vocabulary} tokens, top-k {topK}, top-p {topP}, min-p {minP}: run {run} differs from the first at statistic {at} ({(at < 0 ? 0 : values[at])} against {(at < 0 ? 0 : first[at])})");
                    }
                }
            }
            finally
            {
                x.Release();
                ids.Release();
                stats.Release();
                step.Release();
            }
        }
    }

    // Barriers only between dependent commands: a chain of dependent dispatches gets one before each link and gives the
    // chained result; dispatches on separate storages get none; writing a storage read earlier in the span (write after
    // read) and reading or writing one written in it get one. Checked with pushed descriptors and with descriptor sets.
    private static void VulkanBarriers(Device device)
    {
        List<int> ordinals = device.Type switch
        {
            DeviceType.Vulkan => [device.Ordinal],
            DeviceType.Cpu => [.. Enumerable.Range(0, VulkanBackend.DeviceCount)],
            _ => [],
        };
        foreach (var (ordinal, push) in ordinals.SelectMany(o => new[] { (o, true), (o, false) }))
        {
            VulkanBarriers(ordinal, push);
        }
    }

    // Axpb (VulkanTests.cs) declaring that it writes only y, so dispatches reading the same storages may overlap.
    private static VulkanKernel? s_axpbWritesY;

    private static VulkanKernel AxpbWritesY => s_axpbWritesY ??= new(Axpb.Spirv, bindings: 3, pushConstantBytes: 8, name: "axpb", writes: 0b100);

    private static void VulkanBarriers(int ordinal, bool push)
    {
        {
            var backend = VulkanBackend.CreateSeparate(ordinal, preferMapped: true, pushDescriptors: push);
            string where = $"vulkan:{ordinal}, {(backend.PushDescriptors ? "pushed descriptors" : "descriptor sets")}";

            // y = a · alpha + b over the storages' n values, writing only y.
            void Queue(Storage a, Storage b, Storage y, float alpha)
            {
                var arguments = new AxpbArguments { N = (uint)a.Length, Alpha = alpha };
                backend.Dispatch(AxpbWritesY, (uint)((a.Length + 63) / 64), 1, 1, [a, b, y],
                    System.Runtime.InteropServices.MemoryMarshal.AsBytes(new ReadOnlySpan<AxpbArguments>(in arguments)));
            }

            const int N = 300_000;                                       // enough work per dispatch for overlap to show on a GPU
            var storages = Enumerable.Range(0, 10).Select(_ => backend.Allocate(N, zeroed: true)).ToArray();
            try
            {
                backend.Upload(Enumerable.Repeat(1f, N).ToArray(), storages[0]);
                backend.Synchronize();

                // A chain: s[i + 1] = 2 s[i] + s[i + 1], each link reading the previous link's output (the first one depends
                // on nothing queued).
                long barriers = backend.Barriers;
                for (int i = 0; i < 8; i++)
                {
                    Queue(storages[i], storages[i + 1], storages[i + 1], 2f);
                }

                // Through staging, the upload of s[0] was a copy in the current span, so the first link waits for it too.
                int expected = backend.UnifiedMemory ? 7 : 8;
                Check(backend.Barriers - barriers == expected, $"{where}: a chain of 8 dependent dispatches has {backend.Barriers - barriers} barriers, expected {expected}");
                var values = new float[N];
                backend.Download(storages[8], values);
                Check(values.All(v => v == 256f), $"{where}: the chain's result {values[0]}, expected 256");

                // Independent dispatches: each writes its own storage from one only read.
                backend.Synchronize();
                barriers = backend.Barriers;
                for (int i = 1; i < 9; i++)
                {
                    Queue(storages[0], storages[0], storages[i], 1f);                         // s[i] = 2 s[0], s[0] only read
                }

                // One barrier: before the last, whose storage the chain's last link wrote in the same span.
                Check(backend.Barriers - barriers == 1, $"{where}: 8 independent dispatches had {backend.Barriers - barriers} barriers, expected 1");
                for (int i = 1; i < 9; i++)
                {
                    backend.Download(storages[i], values);
                    Check(values.All(v => v == 2f), $"{where}: independent dispatch {i} gave {values[0]}");
                }

                // Write after read: s[9] reads s[1]; then s[1] is overwritten; s[9] must hold the old value.
                backend.Synchronize();
                barriers = backend.Barriers;
                Queue(storages[1], storages[0], storages[9], 1f);                             // s[9] = s[1] + s[0] = 3: no barrier
                Queue(storages[0], storages[0], storages[1], 5f);                             // s[1] = 6, after the read
                Queue(storages[1], storages[0], storages[2], 1f);                             // s[2] = s[1] + 1 = 7, after the write
                Check(backend.Barriers - barriers == 2, $"{where}: read, then overwritten, then read again: {backend.Barriers - barriers} barriers, expected 2");
                backend.Download(storages[9], values);
                Check(values.All(v => v == 3f), $"{where}: write after read gave {values[0]}, expected 3");
                backend.Download(storages[2], values);
                Check(values.All(v => v == 7f), $"{where}: read after write gave {values[0]}, expected 7");

                // Recording a dispatch allocates nothing.
                backend.Synchronize();
                var argument = new byte[8];
                long allocated = GC.GetAllocatedBytesForCurrentThread();
                for (int i = 0; i < 100; i++)
                {
                    backend.Dispatch(AxpbWritesY, 1, 1, 1, [storages[i % 8], storages[8], storages[9]], argument);
                }

                allocated = GC.GetAllocatedBytesForCurrentThread() - allocated;
                Check(allocated == 0, $"{where}: 100 dispatches allocated {allocated} bytes");
                backend.Synchronize();
            }
            finally
            {
                foreach (var s in storages)
                {
                    s.Release();
                }

                backend.Shutdown();
            }
        }
    }

    // Decoding steps of a decoder (float32, int8, int4 and bfloat16 weights) and of a multi-head attention model, over
    // float32, int8 and bfloat16 caches, with penalties and top-k, top-p and min-p sampling: no step takes the host fallback.
    private static void VulkanDecodingWithoutFallbacks(Device device)
    {
        if (device.Type != DeviceType.Vulkan)
        {
            return;                                                      // run on the Vulkan device itself
        }

        if (Environment.GetEnvironmentVariable("IDRAK_VULKAN_KERNELS") is "0" or "false")
        {
            Console.WriteLine("    (IDRAK_VULKAN_KERNELS=0: kernels off, skipped)");
            return;
        }

        var backend = device.Backend;
        var spec = SmallSpec with { QkNorm = true };
        var models = new List<(string Name, Sequential Model, int Vocabulary)>();
        foreach (var (name, options) in new[]
        {
            ("decoder, float32 weights", new DecoderBuildOptions { Device = device, Seed = 2 }),
            ("decoder, int8 weights", new DecoderBuildOptions { Device = device, Seed = 2, Int8 = true }),
            ("decoder, int4 weights", new DecoderBuildOptions { Device = device, Seed = 2, Int4 = true }),
            ("decoder, bfloat16 weights", new DecoderBuildOptions { Device = device, Seed = 2, BFloat16 = true }),
        })
        {
            models.Add((name, spec.Build(null, options), spec.Vocabulary));
        }

        var (attention, tokenizer) = TinyLanguageModel(device);
        models.Add(("multi-head attention", attention, tokenizer.VocabularySize));
        var counts = new ConcurrentDictionary<string, long>();
        var problems = new List<string>();
        CountHostCalls(backend, counts);
        try
        {
            foreach (var (name, model, vocabulary) in models)
            {
                model.Eval();
                foreach (var format in new[] { KeyValueFormat.Float32, KeyValueFormat.Int8, KeyValueFormat.BFloat16 })
                {
                    using var noGrad = Autograd.NoGrad();
                    using var context = new DecodingContext(device, 1, 32, format) { LastPositionOnly = true };
                    using var sampler = new TokenSampler(device, 1, vocabulary, 32)
                    {
                        Temperature = 0.8f, TopK = 5, TopP = 0.9f, MinP = 0.05f, RepeatPenalty = 1.1f, PresencePenalty = 0.1f, FrequencyPenalty = 0.1f, Seed = 7,
                    };
                    sampler.SetHistory([1, 2, 3]);
                    using (new TensorScope())
                    {
                        sampler.Sample(model.ForwardCached(Tensor.From([1f, 2f, 3f], [1, 3], device), context));
                    }

                    for (int step = 0; step < 6; step++)
                    {
                        if (step == 2)
                        {
                            counts.Clear();                              // after warm-up: the steps alone
                        }

                        using var scope = new TensorScope();
                        sampler.Sample(model.ForwardCached(sampler.Ids.Reshape(1, 1), context));
                    }

                    if (!counts.IsEmpty)
                    {
                        problems.Add($"{name}, {format} cache: "
                            + string.Join(", ", counts.OrderBy(c => c.Key).Select(c => $"{c.Key} ×{c.Value / 4.0}")) + " per step");
                    }

                    var tokens = sampler.Read(0, 7);
                    Check(tokens.All(t => t[0].Id >= 0 && t[0].Id < vocabulary), $"{name}, {format} cache: tokens in range");
                }
            }

            Check(problems.Count == 0, "host fallbacks in decoding steps: " + string.Join("; ", problems));
        }
        finally
        {
            CountHostCalls(backend, null);
            foreach (var (_, model, _) in models)
            {
                model.Dispose();
            }
        }
    }
}
