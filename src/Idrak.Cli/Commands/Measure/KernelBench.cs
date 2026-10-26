// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
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
/// matrix products (float32 and with mixed-precision bfloat16), decoding-sized products of one row with packed weights
/// (int8, int4, bfloat16 or any registered packed format; the built-in ones through the device product the tensor's runs,
/// into one output kept across calls, so a call allocates nothing), and one decoding step of an attention layer over a cache of
/// several lengths in each key/value format. The shapes follow the test runner's --bench-vulkan and --bench-gemv
/// sections, which time the same work through each backend's internals. Each measurement's first call is reported
/// apart (cold) and its value is the median of the timed calls after a warm-up that settles (<see cref="BenchTiming"/>).
/// </summary>
internal static class KernelBench
{
    /// <summary>The benchmark names <c>--kernels</c> accepts.</summary>
    public static readonly string[] Names = ["matmul", "gemv", "attention"];

    /// <summary>Runs the chosen benchmarks on <paramref name="device"/>; <paramref name="small"/> uses small shapes (slow devices, quick checks).</summary>
    public static List<BenchResult> Run(Device device, IReadOnlyCollection<string> kernels, int repeats, bool small, IReadOnlyList<string> packedFormats,
        BenchSettle settle, Action<string>? progress = null, Action<BenchResult>? measured = null)
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
                BenchResult Rate(string name, BenchTime time) => new(name, flops / (time.Micros * 1e3), "GFLOP/s", true,
                    string.Create(CultureInfo.InvariantCulture, $"{time.Micros / 1000:F2} ms"), flops / (time.ColdMicros * 1e3), time.WarmupCalls, time.Settled,
                    time.ColdJitMilliseconds, time.ColdJitMethods);
                Add(Rate($"matmul {size}x{size}x{size} float32", Time(device, Product, repeats, settle)));
                using (MixedPrecision.BFloat16())
                {
                    Add(Rate($"matmul {size}x{size}x{size} mixed bfloat16", Time(device, Product, repeats, settle)));
                }
            }
        }

        if (kernels.Contains("gemv"))
        {
            var shapes = small ? new[] { (256, 768), (768, 256) } : [(1024, 3072), (3072, 1024), (1024, 32_000)];
            // One weights buffer for every shape (refilled for each), not a new array of the largest size a shape.
            var buffer = new float[shapes.Max(s => (long)s.Item1 * s.Item2)];
            foreach (var (k, n) in shapes)
            {
                var weights = buffer.AsSpan(0, k * n);
                for (int i = 0; i < weights.Length; i++)
                {
                    weights[i] = random.NextSingle() - 0.5f;
                }

                using var x = Tensor.From(Values(k), [1, k], device);
                using var y = Tensor.Empty([1, n], device, track: false);
                foreach (string format in packedFormats)
                {
                    progress?.Invoke($"gemv {format} {k}x{n}");
                    using var packed = PackedWeightFormats.Pack(format, weights, k, n, device);
                    var time = packed is IKernelPackedWeight kernel
                        ? Time(device, () => PackedProduct(kernel, x, y), repeats, settle)
                        : Time(device, () =>
                        {
                            using var scope = new TensorScope();
                            packed.MatMul(x);
                        }, repeats, settle);
                    Add(new BenchResult($"gemv {format} 1x{k} -> {n}", time.Micros, "µs", false,
                        string.Create(CultureInfo.InvariantCulture, $"{packed.Bytes / (time.Micros * 1e3):F1} GB/s of weights, {time.AllocatedBytes:F0} B allocated a call"),
                        time.ColdMicros, time.WarmupCalls, time.Settled, time.ColdJitMilliseconds, time.ColdJitMethods));
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
                    // Every call steps from the same `length` cached positions: the cache is cut back to them before each.
                    using var context = new DecodingContext(device, 1, length + 2, KeyValueLayouts.Get(format));
                    using (var fill = Tensor.From(Values((long)length * dim), [1, length, dim], device))
                    {
                        Cached(layer, fill, context);
                    }

                    var time = Time(device, () => Cached(layer, step, context), repeats, settle, () => context.Truncate(length));
                    Add(new BenchResult($"attention decode step {format} cache, {length} positions (dim {dim}, {heads} heads)", time.Micros, "µs", false,
                        null, time.ColdMicros, time.WarmupCalls, time.Settled, time.ColdJitMilliseconds, time.ColdJitMethods));
                }
            }
        }

        return results;

        void Add(BenchResult result)
        {
            results.Add(result);
            measured?.Invoke(result);
        }
    }

    // One row through weights in a built-in packed format into `y` (kept across calls): the device product the tensor's
    // few-row product runs (Tensor.MatMul through IPackedWeight.MatMul: the int8 one checks first whether the device
    // prefers its packed product for the shape), writing the output the bench keeps instead of a new tensor a call, so it
    // times the kernel and what the device does to queue it, not the output tensor a call would make.
    private static void PackedProduct(IKernelPackedWeight weight, Tensor x, Tensor y)
    {
        var backend = x.Backend;
        var (packed, scales) = (weight.PackedValues.Storage, weight.ScaleValues?.Storage);
        int k = weight.Rows, n = weight.Columns;
        switch (weight.Format)
        {
            case PackedFormat.Int8:
                if (!(backend.PrefersPackedMatMul(PackedFormat.Int8, x.Storage, packed, scales, y.Storage, 1, n, k)
                      && backend.PackedMatMulLarge(PackedFormat.Int8, x.Storage, packed, scales, y.Storage, 1, n, k)))
                {
                    backend.Int8MatMul(x.Storage, packed, scales!, y.Storage, 1, n, k);
                }

                break;
            case PackedFormat.Int4:
                backend.Int4MatMul(x.Storage, packed, scales!, y.Storage, 1, n, k);
                break;
            case PackedFormat.BFloat16:
                backend.BFloat16MatMul(x.Storage, packed, y.Storage, 1, n, k);
                break;
            default:
                throw new NotSupportedException($"gemv: no device product for the built-in format {weight.Format}.");
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
