// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak;

// A bidirectional attention pass the size of SigLIP's (4,096 tokens, 16 heads of 72 dimensions; IDRAK_SPAN_TOKENS
// changes the tokens) on each device IDRAK_DEVICES names (default the CPU): through AttentionSpans (tiled, no score
// matrix) or composed (the full [heads, T, T] scores, softmax, values). One path per process (its argument: spans or
// composed, default spans), so the peak memory it prints (VmHWM on Linux) is that path's.
internal static partial class Tests
{
    internal static int BenchSpans(string path)
    {
        int tokens = int.TryParse(Environment.GetEnvironmentVariable("IDRAK_SPAN_TOKENS"), out int t) ? t : 4096, heads = 16, dim = 72;
        List<Device> devices = Environment.GetEnvironmentVariable("IDRAK_DEVICES") is { Length: > 0 } chosen
            ? [.. chosen.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(Device.Parse)]
            : [Device.Cpu];
        float scale = 1f / MathF.Sqrt(dim);
        foreach (var device in devices)
        {
            var random = new Random(3);
            using var q = Tensor.From(RandomArray(random, heads * tokens * dim), [heads, tokens, dim], device);
            using var k = Tensor.From(RandomArray(random, heads * tokens * dim), [heads, tokens, dim], device);
            using var v = Tensor.From(RandomArray(random, heads * tokens * dim), [heads, tokens, dim], device);
            var (starts, ends) = KeySpans.Bidirectional(tokens, tokens).ToTensors(device);
            Action run = path == "composed"
                ? () =>
                {
                    using var scores = q.MatMul(k, transposeB: true);
                    using var weights = scores.ScaleMaskSoftmax(scale, null);
                    using var y = weights.MatMul(v);
                }
                : () =>
                {
                    using var y = Tensor.AttentionSpans(q, k, v, starts, ends, scale);
                };
            using (Autograd.NoGrad())
            {
                run();
                device.Synchronize();
                var watch = Stopwatch.StartNew();
                const int Repeats = 2;
                for (int i = 0; i < Repeats; i++)
                {
                    run();
                }

                device.Synchronize();
                Console.WriteLine($"{device}: {path}, {tokens} tokens, {heads} heads, dim {dim}: {watch.Elapsed.TotalMilliseconds / Repeats:F0} ms a pass, "
                    + $"peak memory {PeakMegabytes()} MB");
            }
        }

        return 0;
    }

    // The process's peak resident memory in MB (Linux; -1 elsewhere).
    private static long PeakMegabytes()
    {
        try
        {
            var line = File.ReadLines("/proc/self/status").FirstOrDefault(l => l.StartsWith("VmHWM:", StringComparison.Ordinal));
            return line is null ? -1 : long.Parse(line.Split(' ', StringSplitOptions.RemoveEmptyEntries)[1]) / 1024;
        }
        catch (IOException)
        {
            return -1;
        }
    }
}
