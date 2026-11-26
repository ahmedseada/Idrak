// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Abstraction.Cluster;
using Idrak.Abstraction.Devices.Cpu;
using Idrak.Audio.Plugins.RnNoise;
using Idrak.Data;
using Idrak.Gpu.Cuda;
using Idrak.Gpu.Hip;
using Idrak.Gpu.Vulkan;
using Idrak.Models;

// The library inside an app (plan 34, M0): one cache folder every cache reads when it first needs a path (an app sets
// its own storage at start-up), never a relative path; progress drawn without a console window; Vulkan's available
// memory within the driver's memory budget where the device reports one; a lost device that names its work.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] AppHostingGroup =
    [
        ("in an app: the progress line draws without a console window (PlatformNotSupportedException) and asks it once; an I/O error is asked again", AppProgressWithoutConsole),
        ("in an app: Vulkan's available memory is the heap's bound, byte for byte, without a memory budget, and at most the budget less the usage plus the cached blocks with one; a lost device names its batch and IDRAK_VULKAN_WIDTH", AppVulkanBudgetAndLostDevice),
    ];

    internal static void AppProgressWithoutConsole(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        int asked = 0;
        var none = new ConsoleWidth(() =>
        {
            asked++;
            throw new PlatformNotSupportedException("no console window on this platform");
        });
        Check(none.Read() == 0 && !none.Supported && none.Read() == 0 && asked == 1,
            $"a platform without a console window: width 0, not supported, asked once ({asked} times)");
        Check(!new ConsoleWidth(() => throw new InvalidOperationException("no console")).Supported, "InvalidOperationException: no console window");

        int io = 0;
        var detached = new ConsoleWidth(() =>
        {
            io++;
            throw new IOException("the handle is invalid");
        });
        Check(detached.Read() == 0 && detached.Supported && detached.Read() == 0 && io == 3, $"an I/O error: width 0, still supported, asked again ({io} times)");
        Check(new ConsoleWidth(() => 80).Read() == 80 && new ConsoleWidth(() => -1).Read() == 0, "a window's width is read as it is, never negative");

        // The "console" reporter (a line redrawn in place) on a writer, with a console that has no window.
        asked = 0;
        var output = new StringWriter();
        var line = new LineProgressReporter(new ProgressRequest("loading", 10, ProgressUnit.Items, output, TimeSpan.Zero), inPlace: true, none);
        for (int i = 1; i <= 10; i++)
        {
            line.Report(i);
        }

        line.Finish("loaded");
        string text = output.ToString();
        Check(text.Contains("loading", StringComparison.Ordinal) && text.Contains("loaded", StringComparison.Ordinal) && asked == 0,
            $"the line draws without a console window and does not ask it again: {text.ReplaceLineEndings(" | ")} (asked {asked} times)");
        Check(ProgressReporters.Choose(new ProgressOptions { Output = output }) == ProgressReporters.QuietReporter, "another writer: quiet, as before");
    }

    internal static void AppVulkanBudgetAndLostDevice(Device device)
    {
        const long MiB = 1L << 20;
        if (device.Type == DeviceType.Cpu)
        {
            // Without a budget, the heap's bound exactly as before: max(0, heap - in use - reserve).
            foreach (var (heap, inUse, reserve) in new[] { (8192 * MiB, 1000 * MiB, 512 * MiB), (1024 * MiB, 900 * MiB, 256 * MiB), (4096 * MiB, 0L, 0L) })
            {
                long before = Math.Max(0, heap - inUse - reserve);
                Check(VulkanBackend.AvailableBytes(heap, inUse, 300 * MiB, reserve, null) == before, $"no budget: {heap}/{inUse}/{reserve} gives the heap's bound {before}");
            }

            // With one: at most the budget less the usage (which counts the cached blocks) plus the cached blocks.
            Check(VulkanBackend.AvailableBytes(8192 * MiB, 1000 * MiB, 200 * MiB, 512 * MiB, (3000 * MiB, 1500 * MiB)) == 1700 * MiB,
                "a budget below the heap's bound: budget - usage + cached");
            Check(VulkanBackend.AvailableBytes(8192 * MiB, 1000 * MiB, 200 * MiB, 512 * MiB, (64000 * MiB, 1500 * MiB)) == 6680 * MiB,
                "a budget above the heap's bound: the heap's bound (the reserve kept)");
            Check(VulkanBackend.AvailableBytes(8192 * MiB, 1000 * MiB, 200 * MiB, 512 * MiB, (1400 * MiB, 1500 * MiB)) == 100 * MiB,
                "over budget: only the cached blocks beyond the excess");
            Check(VulkanBackend.AvailableBytes(8192 * MiB, 1000 * MiB, 50 * MiB, 512 * MiB, (1400 * MiB, 1500 * MiB)) == 0, "far over budget: 0, never negative");

            // A lost device names its work and the ways around it.
            string message = VulkanDriver.DeviceLostMessage("vkWaitForFences", "vulkan:0 (a test device)", 3, "matmul_tiled");
            foreach (string part in new[] { "vkWaitForFences", "VK_ERROR_DEVICE_LOST", "vulkan:0 (a test device)", "a batch of 3 commands", "'matmul_tiled'",
                         "IDRAK_VULKAN_WIDTH", "cpu" })
            {
                Check(message.Contains(part, StringComparison.Ordinal), $"the lost device's message names {part}: {message}");
            }

            using (DeviceException.Handled())
            {
                var lost = Failure<VulkanException>(() => VulkanDriver.Check(VulkanDriver.ErrorDeviceLost, "vkAllocateMemory"));
                Check(lost.Message.Contains("vkAllocateMemory", StringComparison.Ordinal),
                    $"any call that finds the device lost says how to go on: {lost.Message}");
                var other = Failure<VulkanException>(() => VulkanDriver.Check(VulkanDriver.ErrorOutOfDeviceMemory, "vkAllocateMemory"));
                Check(other.Message == "vkAllocateMemory failed: VK_ERROR_OUT_OF_DEVICE_MEMORY (-2)", $"another failure's message as before: {other.Message}");
            }

            return;
        }

        if (device.Backend is not VulkanBackend vulkan)
        {
            return;
        }

        // On a device: the budget read where the device reports it, the heap's bound otherwise.
        var usage = vulkan.GetMemoryUsage();
        long bound = Math.Max(0, vulkan.StorageHeapBytes - usage.InUse - vulkan.MemoryReserve);
        long? available = vulkan.AvailableMemory(), free = vulkan.FreeMemory();
        var budget = vulkan.StorageHeapBudget();
        Console.WriteLine($"    {vulkan.Name}: memory budget {(vulkan.MemoryBudget ? $"read: {budget?.Budget / MiB} MiB, usage {budget?.Usage / MiB} MiB" : "not read")}; " +
                          $"available {available / MiB} MiB, free {(free is { } f ? $"{f / MiB} MiB" : "not reported")}, heap bound {bound / MiB} MiB");
        if (vulkan.MemoryBudget && budget is { } b)
        {
            Check(available is { } a && a >= 0 && a <= bound,
                $"with the budget: available {available} within the heap's bound {bound} (the budget's room now {b.Budget - b.Usage + usage.Cached})");
            Check(free is { } room && room >= 0 && b.Usage > 0, $"with the budget: free {free} reported, usage {b.Usage} counts this process's allocations");
        }
        else
        {
            Check(available == bound && free is null, $"without the budget: available {available} is the heap's bound {bound}, free not reported ({free})");
        }
    }
}
