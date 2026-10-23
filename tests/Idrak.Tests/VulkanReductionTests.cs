// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Gpu.Vulkan;

// A reduction finished in subgroups sums in the order of workgroup memory's tree (KernelBuilder.Reduce): a fused kernel
// and the composed operation it stands for, or a measured variant and another, give the same bits (docs/optimization.md
// rule 86). The subgroup's own reduction summed in an order its driver picked, so an RL update's first ratios were not 1.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] VulkanReductionGroup =
    [
        ("vulkan reductions: reductions finished in subgroups give the bits of reductions through workgroup memory alone (softmax, log-softmax, RMS norm with and without a gain, layer norm fused and for training with its statistics, a whole sum), at the device's width", VulkanReductionOrder),
        ("vulkan reductions: a row's sums have one order fixed by its length: every row kernel gives the same bits at every width the device allows, with and without subgroups, through a workgroup and through one invocation (rows of 1, 40, 1025 and 3001 columns)", VulkanRowOrder),
    ];

    private static void VulkanReductionOrder(Device device)
    {
        List<int> ordinals = device.Type switch
        {
            DeviceType.Vulkan => [device.Ordinal],
            DeviceType.Cpu => [.. Enumerable.Range(0, VulkanBackend.DeviceCount)],
            _ => [],
        };
        if (ordinals.Count == 0)
        {
            if (device.Type == DeviceType.Cpu)
            {
                Console.WriteLine($"    (no Vulkan device: {VulkanBackend.UnavailableReason}; skipped)");
            }

            return;
        }

        if (Environment.GetEnvironmentVariable("IDRAK_VULKAN_KERNELS") is "0" or "false")
        {
            Console.WriteLine("    (IDRAK_VULKAN_KERNELS=0: kernels off, skipped)");
            return;
        }

        const int Rows = 7, Cols = 3001;                                   // longer than any width's narrow rows
        var random = new Random(29);
        float[] R(int n) => [.. Enumerable.Range(0, n).Select(_ => 4 * random.NextSingle() - 2)];
        float[] x = R(Rows * Cols), gain = R(Cols), shift = R(Cols);

        // Every operation's outputs on a backend, by name.
        List<(string Name, float[] Values)> Outputs(VulkanBackend b)
        {
            var made = new List<Storage>();
            Storage Up(float[] values)
            {
                var s = b.Allocate(values.Length, false);
                b.Upload(values, s);
                made.Add(s);
                return s;
            }

            Storage Out(int n)
            {
                var s = b.Allocate(n, true);
                made.Add(s);
                return s;
            }

            float[] Down(Storage s, int n)
            {
                var values = new float[n];
                b.Download(s, values);
                return values;
            }

            try
            {
                var input = Up(x);
                var g = Up(gain);
                var beta = Up(shift);
                var outputs = new List<(string, float[])>();
                foreach (bool log in new[] { false, true })
                {
                    var y = Out(Rows * Cols);
                    b.SoftmaxKernel(input, y, Rows, Cols, log);
                    outputs.Add((log ? "log-softmax" : "softmax", Down(y, Rows * Cols)));
                }

                var norm = Out(Rows * Cols);
                var inv = Out(Rows);
                b.RmsNormKernel(input, norm, inv, Rows, Cols, 1e-6f);
                outputs.Add(("RMS norm", Down(norm, Rows * Cols)));
                outputs.Add(("RMS norm's inverse", Down(inv, Rows)));
                var affine = Out(Rows * Cols);
                b.RmsNormAffineKernel(input, g, affine, Rows, Cols, 1e-6f, 0f);
                outputs.Add(("RMS norm with a gain", Down(affine, Rows * Cols)));
                var fused = Out(Rows * Cols);
                b.LayerNormFusedKernel(input, g, beta, fused, Rows, Cols, 1e-5f);
                outputs.Add(("layer norm", Down(fused, Rows * Cols)));
                var train = Out(Rows * Cols);
                var stats = Out(2 * Rows);
                b.LayerNormTrainKernel(input, g, beta, train, stats, Rows, Cols, 1e-5f);
                outputs.Add(("layer norm for training", Down(train, Rows * Cols)));
                outputs.Add(("layer norm's statistics", Down(stats, 2 * Rows)));
                var total = Out(1);
                b.Sum(input, total, Rows * Cols, 0.5f);
                outputs.Add(("a whole sum", Down(total, 1)));
                return outputs;
            }
            finally
            {
                foreach (var s in made)
                {
                    s.Release();
                }
            }
        }

        bool savedAutotune = VulkanBackend.Autotune;
        VulkanBackend.Autotune = false;                                    // each backend's formula: a workgroup a row here
        try
        {
            foreach (int ordinal in ordinals)
            {
                var subgroups = VulkanBackend.CreateSeparate(ordinal, preferMapped: true);
                List<(string Name, float[] Values)> finished;
                string label;
                try
                {
                    if (!subgroups.Limits.SubgroupReductions)
                    {
                        Console.WriteLine($"    vulkan:{ordinal}: no subgroup shuffles reported: reductions through workgroup memory only (nothing to compare)");
                        continue;
                    }

                    label = $"vulkan:{ordinal} width {subgroups.Width}, butterfly {subgroups.Limits.Butterfly} (subgroups {subgroups.Limits.SubgroupSize})";
                    finished = Outputs(subgroups);
                }
                finally
                {
                    subgroups.Shutdown();
                }

                DeviceLimits.SubgroupsOverride = false;
                VulkanBackend plain;
                try
                {
                    plain = VulkanBackend.CreateSeparate(ordinal, preferMapped: true);
                }
                finally
                {
                    DeviceLimits.SubgroupsOverride = null;
                }

                try
                {
                    Check(plain.Width == subgroups.Width && !plain.Limits.SubgroupReductions, $"{label}: the plain backend has the same width and no subgroup reductions");
                    var tree = Outputs(plain);
                    foreach (var ((name, a), (_, b)) in finished.Zip(tree))
                    {
                        int differ = a.Zip(b).Count(p => BitConverter.SingleToInt32Bits(p.First) != BitConverter.SingleToInt32Bits(p.Second));
                        int first = -1;
                        for (int i = 0; i < a.Length && first < 0; i++)
                        {
                            if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i]))
                            {
                                first = i;
                            }
                        }

                        Check(differ == 0, $"{label}: {name}: {differ} of {a.Length} values differ from workgroup memory's tree; the first, {first}, is {a[Math.Max(0, first)]:R} against {b[Math.Max(0, first)]:R}");
                    }

                    Console.WriteLine($"    {label}: {finished.Count} outputs bit for bit");
                }
                finally
                {
                    plain.Shutdown();
                }
            }
        }
        finally
        {
            VulkanBackend.Autotune = savedAutotune;
        }
    }

    private static void VulkanRowOrder(Device device)
    {
        List<int> ordinals = device.Type switch
        {
            DeviceType.Vulkan => [device.Ordinal],
            DeviceType.Cpu => [.. Enumerable.Range(0, VulkanBackend.DeviceCount)],
            _ => [],
        };
        if (ordinals.Count == 0 || Environment.GetEnvironmentVariable("IDRAK_VULKAN_KERNELS") is "0" or "false")
        {
            if (device.Type == DeviceType.Cpu)
            {
                Console.WriteLine($"    (no Vulkan device or kernels off: {VulkanBackend.UnavailableReason}; skipped)");
            }

            return;
        }

        var random = new Random(31);
        (int Rows, int Cols)[] shapes = [(3, 1), (5, 40), (4, 1025), (7, 3001)];
        var inputs = shapes.Select(s => (Shape: s, X: Enumerable.Range(0, s.Rows * s.Cols).Select(_ => 4 * random.NextSingle() - 2).ToArray(),
            Gain: Enumerable.Range(0, s.Cols).Select(_ => 4 * random.NextSingle() - 2).ToArray(),
            Shift: Enumerable.Range(0, s.Cols).Select(_ => 4 * random.NextSingle() - 2).ToArray())).ToArray();

        // Every row operation's outputs for every shape, by name.
        List<(string Name, float[] Values)> Outputs(VulkanBackend b)
        {
            var outputs = new List<(string, float[])>();
            foreach (var (shape, xs, gains, shifts) in inputs)
            {
                var (rows, cols) = shape;
                var made = new List<Storage>();
                Storage Up(float[] values)
                {
                    var st = b.Allocate(values.Length, false);
                    b.Upload(values, st);
                    made.Add(st);
                    return st;
                }

                Storage Out(int n)
                {
                    var st = b.Allocate(n, true);
                    made.Add(st);
                    return st;
                }

                float[] Down(Storage st, int n)
                {
                    var values = new float[n];
                    b.Download(st, values);
                    return values;
                }

                try
                {
                    var (x, g, beta) = (Up(xs), Up(gains), Up(shifts));
                    string at = $"{rows} x {cols}";
                    foreach (bool log in new[] { false, true })
                    {
                        var y = Out(rows * cols);
                        b.SoftmaxKernel(x, y, rows, cols, log);
                        outputs.Add(($"{(log ? "log-softmax" : "softmax")} {at}", Down(y, rows * cols)));
                    }

                    var (norm, inv) = (Out(rows * cols), Out(rows));
                    b.RmsNormKernel(x, norm, inv, rows, cols, 1e-6f);
                    outputs.Add(($"RMS norm {at}", Down(norm, rows * cols)));
                    outputs.Add(($"RMS norm's inverse {at}", Down(inv, rows)));
                    var affine = Out(rows * cols);
                    b.RmsNormAffineKernel(x, g, affine, rows, cols, 1e-6f, 0f);
                    outputs.Add(($"RMS norm with a gain {at}", Down(affine, rows * cols)));
                    var fused = Out(rows * cols);
                    b.LayerNormFusedKernel(x, g, beta, fused, rows, cols, 1e-5f);
                    outputs.Add(($"layer norm {at}", Down(fused, rows * cols)));
                    var (train, stats) = (Out(rows * cols), Out(2 * rows));
                    b.LayerNormTrainKernel(x, g, beta, train, stats, rows, cols, 1e-5f);
                    outputs.Add(($"layer norm for training {at}", Down(train, rows * cols)));
                    outputs.Add(($"layer norm's statistics {at}", Down(stats, 2 * rows)));
                }
                finally
                {
                    foreach (var st in made)
                    {
                        st.Release();
                    }
                }
            }

            return outputs;
        }

        bool savedAutotune = VulkanBackend.Autotune;
        var savedWidth = DeviceLimits.WidthOverride;
        VulkanBackend.Autotune = false;                                    // the formula: one invocation a row for rows up to a quarter of the width
        try
        {
            foreach (int ordinal in ordinals)
            {
                List<(string Name, float[] Values)>? reference = null;
                string? referenceLabel = null;
                var seen = new List<string>();
                for (int width = VulkanKernels.MinWidth; width <= VulkanKernels.MaxWidth; width *= 2)
                {
                    foreach (bool subgroups in new[] { true, false })
                    {
                        DeviceLimits.WidthOverride = width;
                        DeviceLimits.SubgroupsOverride = subgroups ? null : false;
                        VulkanBackend backend;
                        try
                        {
                            backend = VulkanBackend.CreateSeparate(ordinal, preferMapped: true);
                        }
                        finally
                        {
                            DeviceLimits.SubgroupsOverride = null;
                        }

                        try
                        {
                            if (backend.Width != width || subgroups && !backend.Limits.SubgroupReductions)
                            {
                                continue;                                  // a width past the device's limits, or no subgroups
                            }

                            string label = $"vulkan:{ordinal} width {width}{(subgroups ? $", butterfly {backend.Limits.Butterfly}" : ", workgroup memory")}";
                            var outputs = Outputs(backend);
                            seen.Add(label);
                            if (reference is null)
                            {
                                (reference, referenceLabel) = (outputs, label);
                                continue;
                            }

                            foreach (var ((name, a), (_, b)) in outputs.Zip(reference))
                            {
                                int first = -1, differ = 0;
                                for (int i = 0; i < a.Length; i++)
                                {
                                    if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i]))
                                    {
                                        differ++;
                                        first = first < 0 ? i : first;
                                    }
                                }

                                Check(differ == 0, $"{label}: {name}: {differ} of {a.Length} values differ from {referenceLabel}; the first, {first}, is {a[Math.Max(0, first)]:R} against {b[Math.Max(0, first)]:R}");
                            }
                        }
                        finally
                        {
                            backend.Shutdown();
                        }
                    }
                }

                Check(seen.Count >= 2, $"vulkan:{ordinal}: only {seen.Count} configuration ({string.Join(", ", seen)}) to compare");
                Console.WriteLine($"    vulkan:{ordinal}: {reference!.Count} outputs bit for bit in {seen.Count} configurations ({string.Join("; ", seen)})");
            }
        }
        finally
        {
            VulkanBackend.Autotune = savedAutotune;
            DeviceLimits.WidthOverride = savedWidth;
        }
    }
}
