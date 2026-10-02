// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using Idrak;
using Idrak.Backends;
using Idrak.Backends.Cpu;
using Idrak.Backends.Vulkan;
using Idrak.Layers;
using Idrak.Optimizers;

// Convolutions and pooling on Vulkan: im2col, col2im, max pooling and its gradient as generated kernels, checked against
// the CPU, and a small CNN trained a step with no host fallback. Run on a Vulkan device; elsewhere they do nothing.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] VulkanCnnGroup =
    [
        ("vulkan cnn: im2col, col2im, maxpool and maxpool backward run as kernels and match the CPU bit for bit (stride, padding, odd sizes, batches, channels, ties); group scale, shift and sums too", VulkanCnnOpsMatchCpu),
        ("vulkan cnn: Conv2d, MaxPool2d and GlobalAveragePool2d forward and backward match the CPU with no host fallback", VulkanCnnLayersMatchCpu),
        ("vulkan cnn: a small CNN training step gives the CPU's loss with no host fallback", VulkanCnnTrainingStep),
    ];

    // Window geometries: (N, C, H, W, KH, KW, SH, SW, PH, PW). Odd sizes, strides below, at and above the window,
    // padding, non-square windows, a 1x1 window, a window as large as the padded input.
    private static readonly ConvGeometry[] CnnGeometries =
    [
        new(1, 1, 4, 4, 2, 2, 2, 2, 0, 0),
        new(2, 3, 7, 5, 3, 3, 1, 1, 1, 1),
        new(3, 2, 9, 11, 3, 3, 2, 2, 1, 1),
        new(2, 4, 8, 6, 3, 2, 2, 1, 1, 0),
        new(1, 5, 5, 5, 1, 1, 1, 1, 0, 0),
        new(2, 3, 6, 7, 5, 3, 3, 2, 2, 1),
        new(4, 2, 13, 13, 2, 2, 3, 3, 0, 0),
        new(1, 2, 3, 3, 5, 5, 1, 1, 1, 1),
        new(2, 16, 15, 17, 3, 3, 2, 2, 1, 1),
        new(2, 2, 3, 4, 1, 1, 2, 2, 1, 1),                               // padding >= window: whole windows in the padding
    ];

    private static void VulkanCnnOpsMatchCpu(Device device)
    {
        if (device.Type != DeviceType.Vulkan)
        {
            return;
        }

        ForEachVulkan(device, (backend, label) =>
        {
            var cpu = CpuBackend.Instance;
            var random = new Random(61);
            float[] R(int n) => [.. Enumerable.Range(0, n).Select(_ => random.NextSingle() * 4f - 2f)];

            // Runs op on the CPU and the device on copies of the same inputs; every storage must match (bit for bit by
            // default: the window kernels add in the CPU's order).
            void Case(string what, float[][] inputs, Action<Backend, Storage[]> op, float tolerance = 0f)
            {
                var host = inputs.Select(d => { var st = cpu.Allocate(d.Length, false); cpu.Upload(d, st); return st; }).ToArray();
                var gpu = inputs.Select(d => { var st = backend.Allocate(d.Length, false); backend.Upload(d, st); return st; }).ToArray();
                try
                {
                    long dispatches = backend.Dispatches, fallbacks = backend.HostCalls;
                    op(cpu, host);
                    op(backend, gpu);
                    Check(backend.Dispatches > dispatches, $"{label}: {what} dispatched no kernel");
                    Check(backend.HostCalls == fallbacks, $"{label}: {what} took the host fallback");
                    for (int i = 0; i < inputs.Length; i++)
                    {
                        var (expected, actual) = (new float[inputs[i].Length], new float[inputs[i].Length]);
                        cpu.Download(host[i], expected);
                        backend.Download(gpu[i], actual);
                        for (int j = 0; j < expected.Length && tolerance == 0f; j++)
                        {
                            if (BitConverter.SingleToInt32Bits(expected[j]) != BitConverter.SingleToInt32Bits(actual[j]))
                            {
                                throw new InvalidOperationException($"{label}: {what}, storage {i}[{j}]: {actual[j]}, expected {expected[j]}");
                            }
                        }

                        if (tolerance > 0f)
                        {
                            AssertClose(expected, actual, tolerance, $"{label}: {what}, storage {i}");
                        }
                    }
                }
                finally
                {
                    foreach (var st in host.Concat(gpu))
                    {
                        st.Release();
                    }
                }
            }

            foreach (var g in CnnGeometries)
            {
                string shape = $"{g}";
                int inputs = g.N * g.C * g.H * g.W, columns = g.Positions * g.PatchSize, pooled = g.N * g.C * g.OH * g.OW;
                Case($"im2col {shape}", [R(inputs), R(columns)], (b, s) => b.Im2Col(s[0], s[1], g));
                Case($"col2im {shape}", [R(columns), R(inputs)], (b, s) => b.Col2Im(s[0], s[1], g));

                // Ties (values from a few levels) check that the first maximum in window order wins, as on the CPU.
                var x = random.Next(2) == 0 ? R(inputs) : [.. Enumerable.Range(0, inputs).Select(_ => (float)random.Next(3))];
                Case($"maxpool {shape}", [x, R(pooled), new float[pooled]], (b, s) => b.MaxPool(s[0], s[1], s[2], g));

                // The backward pass takes the argmax indices the forward pass wrote (identical on both, checked above).
                var y = new float[pooled];
                var argmax = new float[pooled];
                var (hx, hy, ha) = (cpu.Allocate(inputs, false), cpu.Allocate(pooled, false), cpu.Allocate(pooled, false));
                cpu.Upload(x, hx);
                cpu.MaxPool(hx, hy, ha, g);
                cpu.Download(ha, argmax);
                foreach (var st in new[] { hx, hy, ha })
                {
                    st.Release();
                }

                Case($"maxpool backward {shape}", [R(pooled), argmax, R(inputs)], (b, s) => b.MaxPoolBackward(s[0], s[1], s[2], g));
            }

            // A NaN wins its window and stays, as the CPU's Max.
            var withNan = new ConvGeometry(1, 1, 4, 4, 2, 2, 2, 2, 0, 0);
            Case("maxpool with NaN", [[1, float.NaN, 2, 0, 3, 4, 8, 1, 9, 2, 6, 7, 0, 1, float.NaN, 2], new float[4], new float[4]],
                (b, s) => b.MaxPool(s[0], s[1], s[2], withNan));

            // Per-group scale and shift (a convolution's bias, its gradient's scaling) and sums (the bias gradient), with
            // and without each optional operand, in place.
            foreach (var (outer, groups, inner) in new[] { (2, 3, 35), (5, 4, 1), (1, 1, 1000), (3, 7, 33) })
            {
                int n = outer * groups * inner;
                string shape = $"[{outer}, {groups}, {inner}]";
                Case($"group scale shift {shape}", [R(n), R(groups), R(groups), R(n)], (b, s) => b.GroupScaleShift(s[0], s[1], s[2], s[3], n, groups, inner, false), 1e-6f);
                Case($"group shift, accumulate {shape}", [R(n), R(groups), R(n)], (b, s) => b.GroupScaleShift(s[0], null, s[1], s[2], n, groups, inner, true), 1e-6f);
                Case($"group scale in place {shape}", [R(n), R(groups)], (b, s) => b.GroupScaleShift(s[0], s[1], null, s[0], n, groups, inner, false), 1e-6f);
                Case($"group reduce {shape}", [R(n), R(n), R(groups), R(groups)], (b, s) => b.GroupReduce(s[0], s[1], s[2], s[3], outer, groups, inner), 1e-4f);
                Case($"group reduce without b {shape}", [R(n), R(groups)], (b, s) => b.GroupReduce(s[0], null, s[1], null, outer, groups, inner), 1e-4f);
            }
        });
    }

    // The device's backend with host calls counted by operation while `run` runs; fails naming the operations that fell back.
    private static void WithoutHostFallback(Device device, string what, Action run)
    {
        var backend = device.Backend;
        var counts = new ConcurrentDictionary<string, long>();
        backend.HostCallsByOperation = counts;
        try
        {
            run();
        }
        finally
        {
            backend.HostCallsByOperation = null;
        }

        Check(counts.IsEmpty, $"{what}: host fallbacks: " + string.Join(", ", counts.OrderBy(c => c.Key).Select(c => $"{c.Key} x{c.Value}")));
    }

    private static void VulkanCnnLayersMatchCpu(Device device)
    {
        if (device.Type != DeviceType.Vulkan)
        {
            return;
        }

        var cpuDevice = Device.Cpu;
        foreach (var (n, c, h, w, oc, k, stride, padding, pool, poolStride, poolPadding) in new[]
        {
            (2, 3, 9, 7, 4, 3, 1, 1, 2, 2, 0),
            (3, 2, 11, 10, 5, 3, 2, 1, 3, 2, 1),
            (1, 4, 6, 6, 3, 1, 1, 0, 2, 1, 0),
            (2, 1, 8, 9, 2, 5, 2, 2, 3, 3, 1),
        })
        {
            string label = $"conv {c}->{oc} k{k} s{stride} p{padding} on {n}x{c}x{h}x{w}, pool {pool}/{poolStride}/{poolPadding}";
            var x = RandomArray(new Random(62), n * c * h * w);
            var conv = new Conv2d(c, oc, k, stride, padding, device: cpuDevice, random: new Random(63));
            var convOnDevice = new Conv2d(c, oc, k, stride, padding, device: device, random: new Random(63));
            var maxPool = new MaxPool2d(pool, poolStride, poolPadding);
            var average = new GlobalAveragePool2d();

            (float[] Output, float[] Pooled, float[] InputGrad, float[] WeightGrad, float[] BiasGrad) Run(Conv2d layer, Device on)
            {
                using var scope = new TensorScope();
                using var input = Tensor.From(x, [n, c, h, w], on, requiresGrad: true);
                var y = layer.Forward(input);
                var pooled = maxPool.Forward(y);
                var averaged = average.Forward(pooled);
                var weights = Tensor.From(RandomArray(new Random(64), averaged.Size), averaged.Shape, on);
                ((averaged * weights).Sum() + (y * y).Sum() * 0.01f).Backward();
                var result = (y.ToArray(), pooled.ToArray(), input.Grad!.ToArray(), layer.Weight.Grad!.ToArray(), layer.Bias!.Grad!.ToArray());
                layer.Weight.ZeroGrad();
                layer.Bias.ZeroGrad();
                return result;
            }

            var expected = Run(conv, cpuDevice);
            var actual = expected;
            WithoutHostFallback(device, label, () => actual = Run(convOnDevice, device));
            AssertClose(expected.Output, actual.Output, 1e-4f, $"{label}: output");
            AssertClose(expected.Pooled, actual.Pooled, 1e-4f, $"{label}: pooled");
            AssertClose(expected.InputGrad, actual.InputGrad, 1e-4f, $"{label}: input gradient");
            AssertClose(expected.WeightGrad, actual.WeightGrad, 1e-3f, $"{label}: weight gradient");
            AssertClose(expected.BiasGrad, actual.BiasGrad, 1e-3f, $"{label}: bias gradient");
            conv.Dispose();
            convOnDevice.Dispose();
        }
    }

    private static void VulkanCnnTrainingStep(Device device)
    {
        if (device.Type != DeviceType.Vulkan)
        {
            return;
        }

        const int N = 4, Classes = 3;
        var random = new Random(65);
        var images = RandomArray(random, N * 2 * 9 * 9);
        var labels = new float[N * Classes];
        for (int i = 0; i < N; i++)
        {
            labels[i * Classes + i % Classes] = 1f;
        }

        Sequential Model(Device on)
        {
            var r = new Random(66);
            return new Sequential
            {
                new Conv2d(2, 4, 3, padding: 1, device: on, random: r), new ReLU(),
                new MaxPool2d(2),
                new Conv2d(4, 6, 3, stride: 2, padding: 1, device: on, random: r), new ReLU(),
                new MaxPool2d(3, 1, padding: 1),
                new GlobalAveragePool2d(),
                new Linear(6, Classes, device: on, random: r),
            };
        }

        // Two steps of SGD with momentum: the losses before each step and after the second.
        float[] Train(Device on)
        {
            using var model = Model(on);
            using var optimizer = new Sgd(model.Parameters(), 0.1f, momentum: 0.9f);
            var losses = new float[3];
            for (int step = 0; step < 3; step++)
            {
                using var scope = new TensorScope();
                using var x = Tensor.From(images, [N, 2, 9, 9], on);
                using var t = Tensor.From(labels, [N, Classes], on);
                var loss = Losses.CrossEntropy(model.Forward(x), t);
                losses[step] = loss.ToArray()[0];
                if (step < 2)
                {
                    optimizer.ZeroGrad();
                    loss.Backward();
                    optimizer.Step();
                }
            }

            return losses;
        }

        var expected = Train(Device.Cpu);
        float[] actual = [];
        WithoutHostFallback(device, "CNN training step", () => actual = Train(device));
        AssertClose(expected, actual, 1e-4f, "CNN losses (before each step, after the second)");
        Check(actual[2] != actual[0], "the steps changed the loss");
    }
}
