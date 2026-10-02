// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using Idrak;
using Idrak.Backends;
using Idrak.Backends.Cpu;
using Idrak.Backends.Vulkan;
using Idrak.Layers;

// Prompt processing on Vulkan: attention over many query rows and the products through packed weights for many rows run
// as kernels (no host fallback) and match the CPU, at the device's workgroup width and at every other width its limits
// allow. Run on a Vulkan device only (other devices pass without checking anything).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] VulkanPromptGroup =
    [
        ("vulkan prompt: tiled attention over many rows matches the CPU as a kernel (head sizes 64, 128, 256 and odd, grouped heads, causal offsets, partial blocks, with and without the log-sum-exp, every width)", VulkanPromptAttention),
        ("vulkan prompt: packed products for many rows (int8, int4, bfloat16) match the CPU as kernels (odd sizes, partial blocks and scale groups; tiled, few-rows, expanded and measured; every width)", VulkanPromptPacked),
        ("vulkan prompt: a decoder's prompt runs without host fallback and matches the CPU (float32, int8, int4, bfloat16 weights; grouped-query attention, head sizes 6 and 64)", VulkanPromptWithoutFallbacks),
    ];

    // Runs `check` on separate backends of the Vulkan device: its own width (full: true), then every other width its
    // limits allow; measured choices go to a file of this test's own, removed afterwards.
    private static void ForEachPromptBackend(Device device, Action<VulkanBackend, string, bool> check)
    {
        if (device.Type != DeviceType.Vulkan)
        {
            return;
        }

        if (Environment.GetEnvironmentVariable("IDRAK_VULKAN_KERNELS") is "0" or "false")
        {
            Console.WriteLine("    (IDRAK_VULKAN_KERNELS=0: kernels off, skipped)");
            return;
        }

        string cache = Path.Combine(Path.GetTempPath(), $"idrak-prompt-tuning-{Environment.ProcessId}.tsv");
        string? savedCache = VulkanBackend.TuningCacheFile;
        var savedWidth = DeviceLimits.WidthOverride;
        VulkanBackend.TuningCacheFile = cache;
        try
        {
            DeviceLimits.WidthOverride = null;
            int deviceWidth = WithBackend(device.Ordinal, mapped: true, b => check(b, Label(b), true));
            for (int width = VulkanKernels.MinWidth; width <= VulkanKernels.MaxWidth; width *= 2)
            {
                if (width != deviceWidth)
                {
                    int w = width;
                    DeviceLimits.WidthOverride = w;
                    WithBackend(device.Ordinal, mapped: true, b =>
                    {
                        if (b.Width == w)
                        {
                            check(b, Label(b), false);
                        }
                    });
                }
            }
        }
        finally
        {
            DeviceLimits.WidthOverride = savedWidth;
            VulkanBackend.TuningCacheFile = savedCache;
            VulkanBackend.TiledAttentionKernel = null;
            VulkanBackend.PackedPromptKernel = null;
            File.Delete(cache);
        }

        string Label(VulkanBackend b) => $"{device} width {b.Width}{(b.Limits.SubgroupArithmetic ? ", subgroups" : "")}";
    }

    // Runs `op` on the CPU and on the device (no host fallback; when given, the kernel `kernel` dispatched) over copies of
    // `inputs`, and checks the outputs (indices in `outputs`) agree within tolerance relative to each output's largest magnitude.
    private static void PromptCase(VulkanBackend backend, string what, float[][] inputs, int[] outputs, Action<Backend, Storage[]> op, string? kernel,
        float tolerance = 1e-4f)
    {
        var expected = RunOn(CpuBackend.Instance, inputs, outputs, op);
        var operations = new ConcurrentDictionary<string, long>();
        var kernels = new ConcurrentDictionary<string, long>();
        (backend.HostCallsByOperation, backend.DispatchesByKernel) = (operations, kernels);
        float[][] actual;
        try
        {
            actual = RunOn(backend, inputs, outputs, op);
        }
        finally
        {
            (backend.HostCallsByOperation, backend.DispatchesByKernel) = (null, null);
        }

        Check(operations.IsEmpty, $"{what}: host fallback for {string.Join(", ", operations.Keys)}");
        Check(kernel is null || kernels.ContainsKey(kernel), $"{what}: {kernel} did not run (ran {string.Join(", ", kernels.Keys)})");
        for (int o = 0; o < outputs.Length; o++)
        {
            float scale = Math.Max(1f, expected[o].Max(MathF.Abs));
            for (int i = 0; i < expected[o].Length; i++)
            {
                if (!(MathF.Abs(expected[o][i] - actual[o][i]) <= tolerance * scale))
                {
                    throw new Exception($"{what}: output {o}, element {i} is {actual[o][i]}, expected {expected[o][i]}");
                }
            }
        }

        static float[][] RunOn(Backend b, float[][] inputs, int[] outputs, Action<Backend, Storage[]> op)
        {
            var storages = inputs.Select(d => { var s = b.Allocate(d.Length, false); b.Upload(d, s); return s; }).ToArray();
            try
            {
                op(b, storages);
                return [.. outputs.Select(o =>
                {
                    var result = new float[inputs[o].Length];
                    b.Download(storages[o], result);
                    return result;
                })];
            }
            finally
            {
                foreach (var s in storages)
                {
                    s.Release();
                }
            }
        }
    }

    private static void VulkanPromptAttention(Device device) => ForEachPromptBackend(device, (backend, label, full) =>
    {
        var random = new Random(31);
        float[] R(int n) => [.. Enumerable.Range(0, n).Select(_ => 2 * random.NextSingle() - 1)];

        // (head size, key/value heads, query heads per key/value head, steps, capacity, position): prompts from position
        // 0 and after earlier steps, limits clamped at the capacity, more rows than a block, odd sizes everywhere.
        var shapes = full
            ? new[] { (64, 2, 1, 20, 64, 0), (128, 2, 2, 17, 80, 5), (33, 3, 4, 9, 40, 11), (256, 1, 1, 13, 30, 17), (7, 2, 3, 40, 100, 50), (128, 1, 1, 70, 300, 200), (1, 1, 2, 5, 6, 0) }
            : [(128, 2, 2, 17, 80, 5), (33, 3, 4, 9, 40, 11)];
        foreach (var (dim, heads, groups, steps, capacity, position) in shapes)
        {
            int rowsPerHead = groups * steps, rows = heads * rowsPerHead, slots = heads * capacity;
            float scale = 1f / MathF.Sqrt(dim);
            foreach (bool? tiled in full ? new bool?[] { true, null } : [true])
            {
                VulkanBackend.TiledAttentionKernel = tiled;
                string shape = $"{label}: head size {dim}, {heads} heads × {groups} groups, {steps} steps, capacity {capacity}, position {position}, {(tiled is null ? "measured" : "tiled")}";
                float[][] inputs = [R(rows * dim), R(slots * dim), R(slots * dim), [position], new float[rows * dim], new float[rows]];
                PromptCase(backend, $"{shape}, with the log-sum-exp", inputs, [4, 5],
                    (b, s) => b.AttentionTiled(s[0], s[1], s[2], s[3], s[4], s[5], heads, rowsPerHead, steps, capacity, dim, scale), "attention_tiled_lse");
                PromptCase(backend, shape, inputs, [4],
                    (b, s) => b.AttentionTiled(s[0], s[1], s[2], s[3], s[4], null, heads, rowsPerHead, steps, capacity, dim, scale), tiled == true ? "attention_tiled" : null);
            }
        }

        VulkanBackend.TiledAttentionKernel = null;
    });

    private static void VulkanPromptPacked(Device device) => ForEachPromptBackend(device, (backend, label, full) =>
    {
        var random = new Random(37);
        float[] R(int n, float lo = -1f, float hi = 1f) => [.. Enumerable.Range(0, n).Select(_ => lo + (hi - lo) * random.NextSingle())];
        float[] Bits(int n) => [.. Enumerable.Range(0, n).Select(_ => BitConverter.Int32BitsToSingle(random.Next() ^ (random.Next() << 16)))];
        float[] BFloat16Pairs(int words) => [.. Enumerable.Range(0, words).Select(_ =>
        {
            uint low = (uint)BitConverter.SingleToInt32Bits(2 * random.NextSingle() - 1) >> 16;
            uint high = (uint)BitConverter.SingleToInt32Bits(2 * random.NextSingle() - 1) & 0xFFFF0000u;
            return BitConverter.Int32BitsToSingle((int)(low | high));
        })];

        // (m, k, n): rows past a block, k not a multiple of the staging step or of an int4 scale group, columns not a
        // multiple of a word or of a block.
        var shapes = full
            ? new[] { (9, 64, 96), (33, 100, 65), (70, 33, 7), (130, 300, 200), (17, 64, 1), (64, 257, 130) }
            : [(33, 100, 65), (70, 33, 7)];
        foreach (var (m, k, n) in shapes)
        {
            foreach (int? kernel in full ? new int?[] { 2, 0, 1, null } : [2])
            {
                VulkanBackend.PackedPromptKernel = kernel;
                string shape = $"{label}: {m}×{k}→{n}, {kernel switch { 0 => "few-rows kernels", 1 => "expanded", 2 => "tiled", _ => "measured" }}";
                int words8 = (n + 3) / 4, words4 = (n + 7) / 8, words2 = (n + 1) / 2;
                float[][] int8 = [R(m * k), Bits(k * words8), R(n, 0.001f, 0.02f), new float[m * n]];
                float[][] int4 = [R(m * k), Bits(k * words4), R((k + 31) / 32 * words4 * 8, 0.01f, 0.2f), new float[m * n]];
                float[][] bf16 = [R(m * k), BFloat16Pairs(k * words2), new float[m * n]];

                // int8 as the tensor calls it: the packed product, else the expanded weights through the float product.
                PromptCase(backend, $"int8 product {shape}", int8, [3], (b, s) =>
                {
                    if (b is CpuBackend)
                    {
                        b.Int8MatMul(s[0], s[1], s[2], s[3], m, n, k);
                    }
                    else if (!b.PackedMatMulLarge(PackedFormat.Int8, s[0], s[1], s[2], s[3], m, n, k))
                    {
                        Check(kernel is 1 or null, $"int8 product {shape}: no packed product");
                        var w = b.Allocate(k * n, false);
                        b.Int8Dequantize(s[1], s[2], w, k, n);
                        b.MatMul(s[0], w, s[3], m, n, k, false, false, 0f);
                        w.Release();
                    }
                }, kernel == 2 ? "int8_gemm" : null);
                PromptCase(backend, $"int4 product {shape}", int4, [3], (b, s) => b.Int4MatMul(s[0], s[1], s[2], s[3], m, n, k), kernel == 2 ? "int4_gemm" : null);
                PromptCase(backend, $"bfloat16 product {shape}", bf16, [2], (b, s) => b.BFloat16MatMul(s[0], s[1], s[2], m, n, k), kernel == 2 ? "bf16_gemm" : null);
                if (kernel == 2)
                {
                    PromptCase(backend, $"int4 packed product {shape}", int4, [3], (b, s) =>
                    {
                        if (b is CpuBackend)
                        {
                            b.Int4MatMul(s[0], s[1], s[2], s[3], m, n, k);
                        }
                        else
                        {
                            Check(b.PackedMatMulLarge(PackedFormat.Int4, s[0], s[1], s[2], s[3], m, n, k), $"int4 packed product {shape}: false");
                        }
                    }, "int4_gemm");
                    PromptCase(backend, $"bfloat16 packed product {shape}", bf16, [2], (b, s) =>
                    {
                        if (b is CpuBackend)
                        {
                            b.BFloat16MatMul(s[0], s[1], s[2], m, n, k);
                        }
                        else
                        {
                            Check(b.PackedMatMulLarge(PackedFormat.BFloat16, s[0], s[1], null, s[2], m, n, k), $"bfloat16 packed product {shape}: false");
                        }
                    }, "bf16_gemm");
                }
            }
        }

        VulkanBackend.PackedPromptKernel = null;
    });

    // A decoder's prompt (12 tokens: tiled attention, products of 12 rows) on the Vulkan device: no host fallback, and
    // the CPU's logits for the same weights.
    private static void VulkanPromptWithoutFallbacks(Device device)
    {
        if (device.Type != DeviceType.Vulkan)
        {
            return;
        }

        if (Environment.GetEnvironmentVariable("IDRAK_VULKAN_KERNELS") is "0" or "false")
        {
            Console.WriteLine("    (IDRAK_VULKAN_KERNELS=0: kernels off, skipped)");
            return;
        }

        int[] ids = [1, 4, 9, 16, 2, 7, 11, 3, 8, 20, 5, 6];
        var specs = new[] { ("head size 6", SmallSpec), ("head size 64", SmallSpec with { Dim = 96, HeadDim = 64, FfDim = 160 }) };
        foreach (var (specName, spec) in specs)
        {
            foreach (var (weights, options) in new (string, Func<Device, DecoderBuildOptions>)[]
            {
                ("float32 weights", d => new DecoderBuildOptions { Device = d }),
                ("int8 weights", d => new DecoderBuildOptions { Device = d, Int8 = true }),
                ("int4 weights", d => new DecoderBuildOptions { Device = d, Int4 = true }),
                ("bfloat16 weights", d => new DecoderBuildOptions { Device = d, BFloat16 = true }),
            })
            {
                string what = $"{specName}, {weights}";
                float[] Prompt(Device d, ConcurrentDictionary<string, long>? counts)
                {
                    using var model = spec.Build(new RandomWeights(71), options(d));
                    model.Eval();
                    using var noGrad = Autograd.NoGrad();
                    using var context = new DecodingContext(d, 1, 32, KeyValueFormat.Float32);
                    using var prompt = Tensor.From([.. ids.Select(i => (float)i)], [1, ids.Length], d);
                    d.Backend.HostCallsByOperation = counts;
                    try
                    {
                        return model.ForwardCached(prompt, context).ToArray();
                    }
                    finally
                    {
                        d.Backend.HostCallsByOperation = null;
                    }
                }

                var expected = Prompt(Device.Cpu, null);
                var counts = new ConcurrentDictionary<string, long>();
                var actual = Prompt(device, counts);
                Check(counts.IsEmpty, $"{what}: host fallback for " + string.Join(", ", counts.OrderBy(c => c.Key).Select(c => $"{c.Key} ×{c.Value}")));
                float range = Math.Max(1f, expected.Max(MathF.Abs));
                AssertClose(expected, actual, 1e-3f * range, $"{what}: prompt logits");
            }
        }
    }
}
