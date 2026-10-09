// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak;
using Idrak.Abstraction.Devices.Cpu;
using Idrak.Gpu.Cuda;
using Idrak.Gpu.Vulkan;

// Convolutions on each device IDRAK_DEVICES names (default every GPU found), in the current precision (IDRAK_MATMUL=bf16
// for the composed path's products on tensor cores or cooperative matrices): for each shape (a backbone's 3 x 3, a stem's
// 7 x 7 stride 2, a pointwise 1 x 1, a depthwise 3 x 3, a text line's tall and wide filters) the forward pass, the input
// gradient and the weight gradient, timed through each path the device has (composed: im2col and its products; tile:
// the implicit product on 16 x 16 tiles; blocked: on 64 x 64 tiles; depthwise), through the measured choice ("auto", the
// one the library runs), and through the host fallback (the CPU's kernel on host copies plus the copies down and up: what
// a device without kernels pays). Milliseconds per call, the median of five timed rounds after a warm-up. On the CPU
// (IDRAK_DEVICES=cpu): the operation as the CPU runs it (its kernel: the composed path, depthwise as direct loops) against
// unfolded patches and products through tensors (Conv2d's path before the operation existed).
internal static partial class Tests
{
    internal static int BenchConv()
    {
        List<Device> devices = Environment.GetEnvironmentVariable("IDRAK_DEVICES") is { Length: > 0 } chosen
            ? [.. chosen.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(Device.Parse)]
            : [.. Device.Available.Where(d => d.Type != DeviceType.Cpu)];
        if (devices.Count == 0)
        {
            Console.WriteLine("No GPU found: name one with IDRAK_DEVICES (cuda:0, vulkan:0).");
            return 1;
        }

        (string Name, ConvGeometry G, int Filters, int Groups)[] shapes =
        [
            ("3x3, 64 -> 64, 56x56, batch 8", new(8, 64, 56, 56, 3, 3, 1, 1, 1, 1), 64, 1),
            ("7x7 stride 2, 3 -> 64, 224x224, batch 8", new(8, 3, 224, 224, 7, 7, 2, 2, 3, 3), 64, 1),
            ("1x1, 256 -> 64, 28x28, batch 8", new(8, 256, 28, 28, 1, 1, 1, 1, 0, 0), 64, 1),
            ("depthwise 3x3, 128 channels, 56x56, batch 8", new(8, 128, 56, 56, 3, 3, 1, 1, 1, 1), 128, 128),
            ("text line 3x3, 1 -> 32, 32x400, batch 16", new(16, 1, 32, 400, 3, 3, 1, 1, 1, 1), 32, 1),
            ("text line 2x1 stride 2x1, 64 -> 128, 16x200, batch 16", new(16, 64, 16, 200, 2, 1, 2, 1, 0, 0), 128, 1),
        ];
        (string Name, int? Path)[] paths = [("host", -1), ("composed", 0), ("tile", 1), ("blocked", 2), ("depthwise", 3), ("auto", null)];
        foreach (var device in devices)
        {
            if (device.Type == DeviceType.Cpu)
            {
                BenchConvCpu(shapes);
                continue;
            }

            var backend = device.Backend;
            Console.WriteLine($"{device} ({backend.Name}): precision {MixedPrecision.Current}; milliseconds per call (forward / input gradient / weight gradient)");
            Console.WriteLine($"{"shape",-56} " + string.Join(" ", paths.Select(p => $"{p.Name,22}")));
            foreach (var (name, g, filters, groups) in shapes)
            {
                var random = new Random(1);
                int input = g.N * g.C * g.H * g.W, output = g.N * filters * g.OH * g.OW, weights = filters * (g.PatchSize / groups);
                using var x = Tensor.From(RandomArray(random, input), [input], device);
                using var w = Tensor.From(RandomArray(random, weights), [weights], device);
                using var dy = Tensor.From(RandomArray(random, output), [output], device);
                using var y = Tensor.Empty([output], device);
                using var dx = Tensor.Empty([input], device, zeroed: true);
                using var dw = Tensor.Empty([weights], device, zeroed: true);
                var cells = new List<string>();
                foreach (var (_, path) in paths)
                {
                    if (path == 3 && !(groups > 1 && groups == g.C))
                    {
                        cells.Add($"{"-",22}");
                        continue;
                    }

                    CudaBackend.ConvolutionPath = path is >= 0 ? path : null;
                    VulkanBackend.ConvolutionPath = path is >= 0 ? path : null;
                    try
                    {
                        double forward = path == -1 ? TimeHost(backend, x.Storage, w.Storage, y.Storage, input, weights, output, b => b.Convolution(Host(b, 0), Host(b, 1), null, Host(b, 2), g, filters, groups, ConvActivation.None))
                            : Time(device, () => backend.Convolution(x.Storage, w.Storage, null, y.Storage, g, filters, groups, ConvActivation.None));
                        double inputGradient = path == -1 ? TimeHost(backend, dy.Storage, w.Storage, dx.Storage, output, weights, input, b => b.ConvolutionBackwardInput(Host(b, 0), Host(b, 1), Host(b, 2), g, filters, groups))
                            : Time(device, () => backend.ConvolutionBackwardInput(dy.Storage, w.Storage, dx.Storage, g, filters, groups));
                        double weightGradient = path == -1 ? TimeHost(backend, x.Storage, dy.Storage, dw.Storage, input, output, weights, b => b.ConvolutionBackwardWeight(Host(b, 0), Host(b, 1), Host(b, 2), g, filters, groups))
                            : Time(device, () => backend.ConvolutionBackwardWeight(x.Storage, dy.Storage, dw.Storage, g, filters, groups));
                        cells.Add($"{$"{forward:F2} / {inputGradient:F2} / {weightGradient:F2}",22}");
                    }
                    catch (Exception e) when (e is not OutOfMemoryException)
                    {
                        cells.Add($"{"failed: " + e.GetType().Name,22}");
                    }
                }

                CudaBackend.ConvolutionPath = null;
                VulkanBackend.ConvolutionPath = null;
                Console.WriteLine($"{name,-56} " + string.Join(" ", cells));
            }
        }

        return 0;
    }

    // The host storages of the current host timing (operands copied down, the result copied up after).
    [ThreadStatic]
    private static Storage[]? t_hostOperands;

    private static Storage Host(Backend backend, int i) => t_hostOperands![i];

    // Median milliseconds of one call: a warm-up (kernels, measured choices), then five rounds of enough calls for ~20 ms.
    private static double Time(Device device, Action run)
    {
        run();
        device.Synchronize();
        var watch = Stopwatch.StartNew();
        run();
        device.Synchronize();
        int repeats = Math.Clamp((int)(20 / Math.Max(watch.Elapsed.TotalMilliseconds, 0.01)), 1, 50);
        var rounds = new double[5];
        for (int r = 0; r < rounds.Length; r++)
        {
            watch.Restart();
            for (int i = 0; i < repeats; i++)
            {
                run();
            }

            device.Synchronize();
            rounds[r] = watch.Elapsed.TotalMilliseconds / repeats;
        }

        Array.Sort(rounds);
        return rounds[2];
    }

    // The host fallback: the two operands copied to host storages, the CPU's kernel, the result copied back up.
    private static double TimeHost(Backend device, Storage a, Storage b, Storage result, int aLength, int bLength, int resultLength, Action<Backend> op)
    {
        var cpu = CpuBackend.Instance;
        var host = new[] { cpu.Allocate(aLength, false), cpu.Allocate(bLength, false), cpu.Allocate(resultLength, true) };
        var buffer = new float[Math.Max(aLength, Math.Max(bLength, resultLength))];
        t_hostOperands = host;
        try
        {
            void Run()
            {
                device.Download(a, buffer.AsSpan(0, aLength));
                cpu.Upload(buffer.AsSpan(0, aLength), host[0]);
                device.Download(b, buffer.AsSpan(0, bLength));
                cpu.Upload(buffer.AsSpan(0, bLength), host[1]);
                op(cpu);
                cpu.Download(host[2], buffer.AsSpan(0, resultLength));
                device.Upload(buffer.AsSpan(0, resultLength), result);
            }

            var watch = Stopwatch.StartNew();
            Run();
            device.Synchronize();
            double once = watch.Elapsed.TotalMilliseconds;
            var rounds = new double[3];
            for (int r = 0; r < rounds.Length && once < 5000; r++)
            {
                watch.Restart();
                Run();
                device.Synchronize();
                rounds[r] = watch.Elapsed.TotalMilliseconds;
            }

            Array.Sort(rounds);
            return once >= 5000 ? once : rounds[1];
        }
        finally
        {
            t_hostOperands = null;
            foreach (var s in host)
            {
                s.Release();
            }
        }
    }

    // The CPU: forward, input and weight gradients through the operation, and the forward pass through tensors (im2col, the
    // product per group and the permutation, as Conv2d ran before).
    private static void BenchConvCpu((string Name, ConvGeometry G, int Filters, int Groups)[] shapes)
    {
        var device = Device.Cpu;
        var backend = device.Backend;
        Console.WriteLine($"cpu ({backend.Name}): milliseconds per call");
        Console.WriteLine($"{"shape",-56} {"operation (fwd / dx / dw)",28} {"tensors (fwd)",14}");
        foreach (var (name, g, filters, groups) in shapes)
        {
            var random = new Random(1);
            int input = g.N * g.C * g.H * g.W, output = g.N * filters * g.OH * g.OW, weights = filters * (g.PatchSize / groups);
            using var x = Tensor.From(RandomArray(random, input), [g.N, g.C, g.H, g.W], device);
            using var w = Tensor.From(RandomArray(random, weights), [filters, g.PatchSize / groups], device);
            using var dy = Tensor.From(RandomArray(random, output), [output], device);
            using var y = Tensor.Empty([output], device);
            using var dx = Tensor.Empty([input], device, zeroed: true);
            using var dw = Tensor.Empty([weights], device, zeroed: true);
            double forward = Time(device, () => backend.Convolution(x.Storage, w.Storage, null, y.Storage, g, filters, groups, ConvActivation.None));
            double inputGradient = Time(device, () => backend.ConvolutionBackwardInput(dy.Storage, w.Storage, dx.Storage, g, filters, groups));
            double weightGradient = Time(device, () => backend.ConvolutionBackwardWeight(x.Storage, dy.Storage, dw.Storage, g, filters, groups));
            double tensors = Time(device, () =>
            {
                using var scope = new TensorScope();
                int positions = g.OH * g.OW, patch = g.PatchSize / groups, perGroup = filters / groups;
                var columns = x.Im2Col(g);
                var grouped = groups == 1 ? columns : columns.Reshape(g.Positions, groups, patch).Permute(1, 0, 2);
                var rows = groups == 1 ? columns.MatMul(w, transposeB: true) : grouped.MatMul(w.Reshape(groups, perGroup, patch), transposeB: true);
                _ = groups == 1 ? rows.Reshape(g.N, positions, filters).Permute(0, 2, 1) : rows.Reshape(groups, g.N, positions, perGroup).Permute(1, 0, 3, 2);
            });
            Console.WriteLine($"{name,-56} {$"{forward:F2} / {inputGradient:F2} / {weightGradient:F2}",28} {tensors,14:F2}");
        }
    }
}
