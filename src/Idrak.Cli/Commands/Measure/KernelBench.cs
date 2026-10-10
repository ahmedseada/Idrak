// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using Idrak.Layers;

namespace Idrak.Cli.Commands.Measure;

/// <summary>One measured result: a name that stays the same between runs (for <c>--compare</c>), a value and its unit.</summary>
/// <param name="Name">What was measured, e.g. "matmul 1024x1024x1024 float32".</param>
/// <param name="Value">The measured value.</param>
/// <param name="Unit">Its unit, e.g. "GFLOP/s".</param>
/// <param name="HigherIsBetter">Whether a larger value is an improvement (false for times and memory).</param>
/// <param name="Note">Extra detail (the time per call, the bandwidth), or null.</param>
internal sealed record BenchResult(string Name, double Value, string Unit, bool HigherIsBetter, string? Note = null);

/// <summary>
/// The kernel benchmarks of <c>idrak bench</c> without a model, through the library's public tensor API on any device:
/// matrix products (float32 and with mixed-precision bfloat16), decoding-sized products of one row with packed weights
/// (int8, int4, bfloat16 or any registered packed format), and one decoding step of an attention layer over a cache of
/// several lengths in each key/value format. The shapes follow the test runner's --bench-vulkan and --bench-gemv
/// sections, which time the same work through each backend's internals.
/// </summary>
internal static class KernelBench
{
    /// <summary>The benchmark names <c>--kernels</c> accepts.</summary>
    public static readonly string[] Names = ["matmul", "gemv", "attention"];

    /// <summary>Runs the chosen benchmarks on <paramref name="device"/>; <paramref name="small"/> uses small shapes (slow devices, quick checks).</summary>
    public static List<BenchResult> Run(Device device, IReadOnlyCollection<string> kernels, int repeats, bool small, IReadOnlyList<string> packedFormats,
        Action<string>? progress = null)
    {
        var results = new List<BenchResult>();
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

        using var noGrad = Autograd.NoGrad();
        if (kernels.Contains("matmul"))
        {
            foreach (int size in small ? new[] { 128, 256 } : [1024, 2048])
            {
                progress?.Invoke($"matmul {size}");
                using var a = Tensor.From(Values((long)size * size), [size, size], device);
                using var b = Tensor.From(Values((long)size * size), [size, size], device);
                void Product()
                {
                    using var scope = new TensorScope();
                    a.MatMul(b);
                }

                double flops = 2.0 * size * size * size;
                double us = Micros(device, Product, repeats);
                results.Add(new BenchResult($"matmul {size}x{size}x{size} float32", flops / (us * 1e3), "GFLOP/s", true, string.Create(CultureInfo.InvariantCulture, $"{us / 1000:F2} ms")));
                using (MixedPrecision.BFloat16())
                {
                    double mixed = Micros(device, Product, repeats);
                    results.Add(new BenchResult($"matmul {size}x{size}x{size} mixed bfloat16", flops / (mixed * 1e3), "GFLOP/s", true, string.Create(CultureInfo.InvariantCulture, $"{mixed / 1000:F2} ms")));
                }
            }
        }

        if (kernels.Contains("gemv"))
        {
            foreach (var (k, n) in small ? new[] { (256, 768), (768, 256) } : [(1024, 3072), (3072, 1024), (1024, 32_000)])
            {
                var weights = Values((long)k * n);
                using var x = Tensor.From(Values(k), [1, k], device);
                foreach (string format in packedFormats)
                {
                    progress?.Invoke($"gemv {format} {k}x{n}");
                    using var packed = PackedWeight.FromValues(format, weights, k, n, device);
                    double us = Micros(device, () =>
                    {
                        using var scope = new TensorScope();
                        packed.MatMul(x);
                    }, repeats);
                    results.Add(new BenchResult($"gemv {format} 1x{k} -> {n}", us, "µs", false, string.Create(CultureInfo.InvariantCulture, $"{packed.Bytes / (us * 1e3):F1} GB/s of weights")));
                }
            }
        }

        if (kernels.Contains("attention"))
        {
            var (dim, heads, lengths) = small ? (128, 2, new[] { 64, 256 }) : (1024, 8, new[] { 512, 2048 });
            using var layer = new MultiHeadAttention(dim, heads, causal: true, device: device, random: new Random(1));
            layer.Eval();
            using var step = Tensor.From(Values(dim), [1, 1, dim], device);
            foreach (string format in new[] { "float32", "int8", "bfloat16" })
            {
                foreach (int length in lengths)
                {
                    progress?.Invoke($"attention {format} {length}");
                    using var context = new DecodingContext(device, 1, length + repeats + 2, KeyValueLayouts.Get(format));
                    using (var fill = Tensor.From(Values((long)length * dim), [1, length, dim], device))
                    {
                        Cached(layer, fill, context);
                    }

                    double us = Micros(device, () => Cached(layer, step, context), repeats);
                    results.Add(new BenchResult($"attention decode step {format} cache, {length} positions (dim {dim}, {heads} heads)", us, "µs", false));
                }
            }
        }

        return results;
    }

    // One cached pass of the attention layer over `input`'s positions.
    private static void Cached(MultiHeadAttention layer, Tensor input, DecodingContext context)
    {
        using var scope = new TensorScope();
        context.BeginStep(input.Shape[1]);
        layer.ForwardCached(input, context);
        context.EndStep(input.Shape[1]);
    }

    /// <summary>Average time of <paramref name="run"/> in µs over <paramref name="repeats"/> calls, after one warm-up, the device drained before and after.</summary>
    public static double Micros(Device device, Action run, int repeats)
    {
        run();
        device.Synchronize();
        var watch = Stopwatch.StartNew();
        for (int i = 0; i < repeats; i++)
        {
            run();
        }

        device.Synchronize();
        return watch.Elapsed.TotalMilliseconds * 1000 / Math.Max(1, repeats);
    }
}
