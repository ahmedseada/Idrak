// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Abstraction.Devices;
using Idrak.Layers;

namespace Idrak.Cli.Commands.Measure;

/// <summary>One measured result: a name that stays the same between runs (for <c>--compare</c>), a value and its unit.</summary>
/// <param name="Name">What was measured, e.g. "matmul 1024x1024x1024 float32".</param>
/// <param name="Value">The measured value.</param>
/// <param name="Unit">Its unit, e.g. "GFLOP/s".</param>
/// <param name="HigherIsBetter">Whether a larger value is an improvement (false for times and memory).</param>
/// <param name="Note">Extra detail (the time per call, the bandwidth), or null.</param>
/// <param name="Cold">The first call's value in the same unit (a kernel benchmark), or null; <paramref name="Value"/> is then the warm median.</param>
/// <param name="WarmupCalls">The warm-up calls before the timed ones (a kernel benchmark), or null.</param>
/// <param name="Settled">Whether the warm-up settled before its budget (a kernel benchmark), or null.</param>
/// <param name="ColdJitMs">The time the JIT spent compiling during the first call, in ms (a kernel benchmark), or null.</param>
/// <param name="ColdJitMethods">The methods the JIT compiled during the first call (a kernel benchmark), or null.</param>
internal sealed record BenchResult(string Name, double Value, string Unit, bool HigherIsBetter, string? Note = null, double? Cold = null,
    int? WarmupCalls = null, bool? Settled = null, double? ColdJitMs = null, long? ColdJitMethods = null);

/// <summary>
/// The kernel benchmarks of <c>idrak bench</c> without a model, through the library's public tensor API on any device:
/// float32 matrix products and one decoding step of an attention layer over a cache of several lengths. The shapes follow the test runner's --bench-vulkan and --bench-gemv
/// sections, which time the same work through each backend's internals. Each measurement's first call is reported
/// apart (cold) and its value is the median of the timed calls after a warm-up that settles (<see cref="BenchTiming"/>).
/// </summary>
internal static class KernelBench
{
    /// <summary>The benchmark names <c>--kernels</c> accepts.</summary>
    public static readonly string[] Names = ["matmul", "attention"];

    /// <summary>The benchmarks run when <c>--kernels</c> names none.</summary>
    public static readonly string[] DefaultNames = ["matmul", "attention"];

    /// <summary>Runs the chosen benchmarks on <paramref name="device"/>; <paramref name="small"/> uses small shapes (slow devices, quick checks).</summary>
    public static List<BenchResult> Run(Device device, IReadOnlyCollection<string> kernels, int repeats, bool small, BenchSettle settle, Action<string>? progress = null, Action<BenchResult>? measured = null)
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
            ProductRoutes.ForgetChoices();
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
                BenchResult Rate(string name, BenchTime time) => new(name, flops / (time.Micros * 1e3), "GFLOP/s", true,
                    string.Create(CultureInfo.InvariantCulture, $"{time.Micros / 1000:F2} ms"), flops / (time.ColdMicros * 1e3), time.WarmupCalls, time.Settled,
                    time.ColdJitMilliseconds, time.ColdJitMethods);
                Add(Rate($"matmul {size}x{size}x{size} float32", Time(device, Product, repeats, settle)));
            }
        }

        if (kernels.Contains("attention"))
        {
            var (dim, heads, lengths) = small ? (128, 2, new[] { 64, 256 }) : (1024, 8, new[] { 512, 2048 });
            using var layer = new MultiHeadAttention(dim, heads, causal: true, device: device, random: new Random(1));
            layer.Eval();
            using var step = Tensor.From(Values(dim), [1, 1, dim], device);
            foreach (int length in lengths)
            {
                progress?.Invoke($"attention {length}");
                // Every call steps from the same `length` cached positions: the cache is cut back to them before each.
                using var context = new DecodingContext(device, 1, length + 2);
                using (var fill = Tensor.From(Values((long)length * dim), [1, length, dim], device))
                {
                    Cached(layer, fill, context);
                }

                var time = Time(device, () => Cached(layer, step, context), repeats, settle, () => context.Truncate(length));
                Add(new BenchResult($"attention decode step float32 cache, {length} positions (dim {dim}, {heads} heads)", time.Micros, "µs", false,
                    null, time.ColdMicros, time.WarmupCalls, time.Settled, time.ColdJitMilliseconds, time.ColdJitMethods));
            }
        }

        return results;

        void Add(BenchResult result)
        {
            results.Add(result);
            measured?.Invoke(result);
        }
    }

    // One cached pass of the attention layer over `input`'s positions.
    private static void Cached(MultiHeadAttention layer, Tensor input, DecodingContext context)
    {
        using var scope = new TensorScope();
        using var step = context.OpenStep(input.Shape[1]);
        layer.ForwardCached(input, context);
        step.Complete();
    }

    /// <summary>
    /// <paramref name="run"/> on <paramref name="device"/>: its first call (cold) and the median of
    /// <paramref name="repeats"/> calls after a warm-up that settles (<see cref="BenchTiming.Measure"/>), each call
    /// timed alone with the device drained before and after it.
    /// </summary>
    public static BenchTime Time(Device device, Action run, int repeats, BenchSettle settle, Action? between = null) =>
        BenchTiming.Measure(run, device.Synchronize, repeats, settle, between);
}
