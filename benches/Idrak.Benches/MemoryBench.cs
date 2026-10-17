// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak.Layers;

namespace Idrak.Benches;

/// <summary>
/// Stress: the memory pool over thousands of allocations of varied shapes (rule 77: device memory does not grow with use;
/// what the pool holds cached against what is in use), a long sequence's attention, and the largest batch of a convolution
/// network's training step the device takes before it runs out (what a run may use). Each part runs within
/// IDRAK_BENCH_SECONDS.
/// </summary>
internal static class MemoryBench
{
    private const string Section = "memory";

    public static void Run(Device device, Settings settings, Report report)
    {
        VariedShapes(device, settings, report);
        LongSequence(device, settings, report);
        LargestBatch(device, settings, report);
    }

    // Shapes that never repeat exactly (lines of other widths, page after page): the pool must serve them from what it
    // holds instead of growing. Reports in use, cached and the peak after the run, and the allocation rate.
    private static void VariedShapes(Device device, Settings settings, Report report)
    {
        var random = new Random(23);
        ComputeResources.ReleaseCachedMemory(device);
        long baseline = ComputeResources.GetMemoryUsage(device).Reserved;
        ComputeResources.ResetPeakMemoryUsage(device);
        var watch = Stopwatch.StartNew();
        long allocations = 0, bytes = 0;
        double limit = Math.Min(settings.Seconds / 3, 20);
        int most = settings.Size == BenchSize.Small ? 1 << 18 : 1 << 22;
        while (watch.Elapsed.TotalSeconds < limit && allocations < 20_000)
        {
            using var scope = new TensorScope();
            var live = new List<Tensor>();
            for (int i = 0; i < 16; i++)
            {
                int n = random.Next(1024, most);
                live.Add(Tensor.Zeros([n], device));
                bytes += 4L * n;
                allocations++;
            }

            using var scaled = live[0] * 2f;                                    // a kernel between the allocations, as a step has
            device.Backend.Synchronize();
        }

        var usage = ComputeResources.GetMemoryUsage(device);
        long peak = usage.Peak;
        string label = "varied shapes";
        Console.WriteLine(Bench.Line(label, ("allocations", allocations.ToString()), ("MiB asked", Bench.F(bytes / 1048576.0, "0")), ("cached after", Bench.Mib(usage.Cached)),
            ("held over start", Bench.Mib(Math.Max(0, usage.Reserved - baseline))), ("peak", Bench.Mib(peak)), ("µs/alloc", Bench.F(watch.Elapsed.TotalMilliseconds * 1000 / allocations, "0.0"))));
        report.Add(Section, label, device, "cached after", usage.Cached / 1048576.0, "MiB");
        report.Add(Section, label, device, "held over start", Math.Max(0, usage.Reserved - baseline) / 1048576.0, "MiB");
        report.Add(Section, label, device, "allocation", watch.Elapsed.TotalMilliseconds * 1000 / allocations, "µs");
        ComputeResources.ReleaseCachedMemory(device);
    }

    // Attention over a long sequence (the composed path's scores are T x T a head): time and peak memory as T grows.
    private static void LongSequence(Device device, Settings settings, Report report)
    {
        var random = new Random(29);
        int[] lengths = settings.Size switch { BenchSize.Small => [512, 1024], BenchSize.Medium => [1024, 4096], _ => [4096, 8192, 16384] };
        const int Heads = 8, Dim = 64;
        foreach (int t in lengths)
        {
            using var q = Bench.RandomTensor(random, [Heads, t, Dim], device, 0.5f);
            using var k = Bench.RandomTensor(random, [Heads, t, Dim], device, 0.5f);
            using var v = Bench.RandomTensor(random, [Heads, t, Dim], device, 0.5f);
            using var starts = Tensor.Zeros([t], device);
            using var ends = Tensor.From([.. Enumerable.Range(1, t).Select(i => (float)i)], [t], device);
            float scale = 1f / MathF.Sqrt(Dim);
            Action attention = () => Tensor.AttentionFastest(q, k, v, starts, ends, scale, everyKey: false);
            var (_, warm) = Bench.Time(device, attention, Math.Min(settings.Repeats, 3), Math.Min(settings.Warmup, 3));
            long peak = Bench.PeakBytes(device, attention);
            string label = $"causal attention, {Heads} heads x {t} x {Dim}";
            Console.WriteLine(Bench.Line(label, ("ms warm", Bench.F(warm, "0.0")), ("peak", Bench.Mib(peak))));
            report.Add(Section, label, device, "warm", warm, "ms");
            report.Add(Section, label, device, "peak", peak / 1048576.0, "MiB");
        }
    }

    // A small convolution network's training step at doubling batches until the device refuses: the peak each batch takes,
    // and the largest that fits (a run may use that; the trainers' out-of-memory halving starts from it).
    private static void LargestBatch(Device device, Settings settings, Report report)
    {
        var random = new Random(31);
        int side = settings.Size == BenchSize.Small ? 32 : 64, channels = settings.Size == BenchSize.Small ? 16 : 64;
        using var network = new Sequential(
            new Conv2d(3, channels, 3, 1, 1, device: device, random: random), new ReLU(),
            new Conv2d(channels, channels, 3, 1, 1, device: device, random: random), new ReLU(),
            new Conv2d(channels, channels, 3, 2, 1, device: device, random: random), new ReLU(),
            new Conv2d(channels, 10, side / 2, device: device, random: random));
        network.Train();
        using var optimizer = new AdamW(network.Parameters(), 1e-3f);
        var watch = Stopwatch.StartNew();
        int largest = 0;
        for (int batch = 8; batch <= 1 << 16 && watch.Elapsed.TotalSeconds < settings.Seconds / 3; batch *= 2)
        {
            try
            {
                long peak;
                double ms;
                using (var input = Bench.RandomTensor(random, [batch, 3, side, side], device, 1f))
                {
                    void Step()
                    {
                        optimizer.ZeroGrad();
                        using var loss = network.Forward(input).Reshape(batch, 10).Square().Mean();
                        loss.Backward();
                        optimizer.Step();
                    }

                    (_, ms) = Bench.Time(device, Step, 1);
                    peak = Bench.PeakBytes(device, Step);
                }

                largest = batch;
                string label = $"conv net step, batch {batch}";
                Console.WriteLine(Bench.Line(label, ("ms warm", Bench.F(ms, "0.0")), ("peak", Bench.Mib(peak)), ("images/s", Bench.F(batch / (ms / 1000), "0"))));
                report.Add(Section, label, device, "warm", ms, "ms");
                report.Add(Section, label, device, "peak", peak / 1048576.0, "MiB");
            }
            catch (ResourceLimitExceededException e)
            {
                Console.WriteLine($"   batch {batch}: out of device memory ({e.Message.Split(':')[0]})");
                ComputeResources.ReleaseCachedMemory(device);
                break;
            }
        }

        Console.WriteLine(Bench.Line("largest conv net batch that fit", ("images", largest.ToString())));
        report.Add(Section, "largest conv net batch", device, "batch", largest, "images");
    }
}
