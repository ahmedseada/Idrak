// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak;
using Idrak.Abstraction.Operations;

// A bidirectional attention pass the size of SigLIP's (4,096 tokens, 16 heads of 72 dimensions; IDRAK_SPAN_TOKENS
// changes the tokens) on each device IDRAK_DEVICES names (default the CPU), in the current precision (IDRAK_MATMUL=bf16
// for tensor cores): through AttentionSpans (tiled, no score matrix), composed (the full [heads, T, T] scores, softmax,
// values) and Tensor.AttentionFastest (the device's measured choice; which one it took is printed). The argument picks
// one (spans, composed or fastest); without it all three run. Each path's peak is the device's own count of the bytes
// its tensors held at once (MemoryUsage.Peak, reset before the path), on every device alike; the process's peak resident
// memory (VmHWM, Linux) is printed too, for the CPU. With "train", each pass is a training step's: the forward pass with
// the gradient recorded and the backward pass (AttentionSpansBackward on the key-range path; the products', the
// softmax's and the scale's gradients on the composed one; the measured choice timed with its gradient).
internal static partial class Tests
{
    internal static int BenchSpans(string path, bool training = false)
    {
        int tokens = int.TryParse(Environment.GetEnvironmentVariable("IDRAK_SPAN_TOKENS"), out int t) ? t : 4096, heads = 16, dim = 72;
        List<Device> devices = Environment.GetEnvironmentVariable("IDRAK_DEVICES") is { Length: > 0 } chosen
            ? [.. chosen.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(Device.Parse)]
            : [Device.Cpu];
        string[] paths = path == "all" ? ["spans", "composed", "fastest"] : [path];
        float scale = 1f / MathF.Sqrt(dim);
        foreach (var device in devices)
        {
            var random = new Random(3);
            using var q = Tensor.From(RandomArray(random, heads * tokens * dim), [heads, tokens, dim], device);
            using var k = Tensor.From(RandomArray(random, heads * tokens * dim), [heads, tokens, dim], device);
            using var v = Tensor.From(RandomArray(random, heads * tokens * dim), [heads, tokens, dim], device);
            var (starts, ends) = KeySpans.Bidirectional(tokens, tokens).ToTensors(device);
            Console.WriteLine($"{device} ({device.Backend.Name}): {tokens} tokens, {heads} heads, dim {dim}, precision {MixedPrecision.Current}, "
                + $"attention path {AttentionPaths.Forced}, available memory {Megabytes(device.Backend.AvailableMemory())} MB{(training ? ", forward and backward" : "")}");
            q.RequiresGrad = k.RequiresGrad = v.RequiresGrad = training;
            using var w = training ? Tensor.From(RandomArray(random, heads * tokens * dim), [heads, tokens, dim], device) : null;
            foreach (string which in paths)
            {
                Action run = training ? () => TrainingPass(which, q, k, v, w!, starts, ends, scale) : which switch
                {
                    "composed" => () =>
                    {
                        using var scores = q.MatMul(k, transposeB: true);
                        using var weights = scores.ScaleMaskSoftmax(scale, null);
                        using var y = weights.MatMul(v);
                    },
                    "fastest" => () =>
                    {
                        using var y = Tensor.AttentionFastest(q, k, v, starts, ends, scale, everyKey: true);
                    },
                    _ => () =>
                    {
                        using var y = Tensor.AttentionSpans(q, k, v, starts, ends, scale);
                    },
                };
                using (training ? null : (IDisposable)Autograd.NoGrad())
                {
                    ComputeResources.ResetPeakMemoryUsage(device);
                    string took = "";
                    run();                                                        // loads kernels, measures the device's choices
                    device.Synchronize();
                    if (which == "fastest")
                    {
                        using var trace = Kernels.Trace(device.Backend);
                        run();
                        device.Synchronize();
                        took = trace.Calls(Ops.AttentionSpans) > 0 ? " (chose AttentionSpans)" : " (chose composed)";
                        if (training)
                        {
                            took += trace.HostCallsByOperation.ContainsKey("AttentionSpansBackward") ? " (its gradient on the host)" : "";
                        }
                    }

                    var watch = Stopwatch.StartNew();
                    const int Repeats = 3;
                    for (int i = 0; i < Repeats; i++)
                    {
                        run();
                    }

                    device.Synchronize();
                    var usage = ComputeResources.GetMemoryUsage(device);
                    Console.WriteLine($"  {which}{took}: {watch.Elapsed.TotalMilliseconds / Repeats:F1} ms a pass, peak device memory {usage.Peak >> 20} MB "
                        + $"(inputs {(4L * heads * tokens * dim * 3 + 8L * tokens) >> 20} MB), process peak {PeakMegabytes()} MB");
                }
            }

            if (!training && device.Type == DeviceType.Cuda && MixedPrecision.UsesTensorCores)
            {
                BenchSpanKernels(q, k, v, starts, ends, scale);
            }
        }

        return 0;
    }

    // Each tensor-core span kernel CUDA can choose for this shape (the single-stage one and the pipelined one, 4 and 8
    // warps a block), forced in turn: ms a pass, and whether its output equals the first one's bit for bit.
    private static void BenchSpanKernels(Tensor q, Tensor k, Tensor v, Tensor starts, Tensor ends, float scale)
    {
        float[]? first = null;
        var previous = Idrak.Gpu.Cuda.CudaBackend.SpanTensorForced;
        try
        {
            foreach (int candidate in Idrak.Gpu.Cuda.PtxKernels.SpanTensorWarps.Concat(Idrak.Gpu.Cuda.PtxKernels.SpanTensorWarps.Select(w => Idrak.Gpu.Cuda.CudaBackend.SpanPipelined + w)))
            {
                Idrak.Gpu.Cuda.CudaBackend.SpanTensorForced = candidate;
                using (Autograd.NoGrad())
                {
                    using var trace = Kernels.Trace(q.Device.Backend);
                    var output = Tensor.AttentionSpans(q, k, v, starts, ends, scale);
                    var values = output.ToArray();
                    output.Dispose();
                    var watch = Stopwatch.StartNew();
                    const int Repeats = 5;
                    for (int i = 0; i < Repeats; i++)
                    {
                        using var y = Tensor.AttentionSpans(q, k, v, starts, ends, scale);
                    }

                    q.Device.Synchronize();
                    first ??= values;
                    bool same = values.AsSpan().SequenceEqual(first);
                    string kernel = candidate >= Idrak.Gpu.Cuda.CudaBackend.SpanPipelined
                        ? $"pipelined, {candidate - Idrak.Gpu.Cuda.CudaBackend.SpanPipelined} warps"
                        : $"single-stage, {candidate} warps";
                    Console.WriteLine($"  tensor-core spans, {kernel}: {watch.Elapsed.TotalMilliseconds / Repeats:F2} ms a pass, "
                        + $"{(same ? "the same output bit for bit" : $"differs from the first by up to {values.Zip(first).Max(p => Math.Abs(p.First - p.Second)):G3}")}"
                        + $"{(trace.HostCallsByOperation.Count > 0 ? ", host fallbacks: " + string.Join(", ", trace.HostCallsByOperation.Keys) : "")}");
                }
            }
        }
        finally
        {
            Idrak.Gpu.Cuda.CudaBackend.SpanTensorForced = previous;
        }
    }

    // One training pass of attention over q, k, v (which take the gradient): forward with the gradient recorded, then
    // backward from Σ y ∘ w (the gradients add into q's, k's and v's); the pass's tensors are released after.
    private static void TrainingPass(string which, Tensor q, Tensor k, Tensor v, Tensor w, Tensor starts, Tensor ends, float scale)
    {
        var made = new List<Tensor>();
        Tensor Keep(Tensor t)
        {
            made.Add(t);
            return t;
        }

        try
        {
            var y = which switch
            {
                "composed" => Keep(Keep(Keep(Keep(q.MatMul(k, transposeB: true)) * scale).Softmax()).MatMul(v)),
                "fastest" => Keep(Tensor.AttentionFastest(q, k, v, starts, ends, scale, everyKey: true)),
                _ => Keep(Tensor.AttentionSpans(q, k, v, starts, ends, scale)),
            };
            Keep(Keep(y * w).Sum()).Backward();
        }
        finally
        {
            foreach (var t in made)
            {
                t.Dispose();
            }
        }
    }

    private static string Megabytes(long? bytes) => bytes is { } b ? (b >> 20).ToString(System.Globalization.CultureInfo.InvariantCulture) : "unreported";

    // The process's peak resident memory in MB (VmHWM on Linux, the peak working set elsewhere).
    private static long PeakMegabytes()
    {
        try
        {
            var line = File.Exists("/proc/self/status")
                ? File.ReadLines("/proc/self/status").FirstOrDefault(l => l.StartsWith("VmHWM:", StringComparison.Ordinal))
                : null;
            if (line is not null)
            {
                return long.Parse(line.Split(' ', StringSplitOptions.RemoveEmptyEntries)[1]) / 1024;
            }
        }
        catch (IOException)
        {
        }
        using var process = System.Diagnostics.Process.GetCurrentProcess();
        return process.PeakWorkingSet64 >> 20;
    }
}
