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
        ("in an app: one cache folder (ComputeResources.CacheFolder, else IDRAK_CACHE) is read when each cache first needs a path (downloads, gguf, CPU, Vulkan, CUDA, HIP, cluster, features, rnnoise), never relative, and each cache's own variable still wins", AppCacheFolder),
        ("in an app: the progress line draws without a console window (PlatformNotSupportedException) and asks it once; an I/O error is asked again", AppProgressWithoutConsole),
        ("in an app: Vulkan's available memory is the heap's bound, byte for byte, without a memory budget, and at most the budget less the usage plus the cached blocks with one; a lost device names its batch, the host retry and IDRAK_VULKAN_WIDTH", AppVulkanBudgetAndLostDevice),
    ];

    internal static void AppCacheFolder(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;                                                              // settings, not devices: once
        }

        string[] variables = ["IDRAK_CACHE", "IDRAK_TUNING_CACHE", "IDRAK_CPU_TUNING_FILE", "IDRAK_VULKAN_TUNING_CACHE", "IDRAK_HIP_KERNEL_CACHE"];
        var saved = variables.Select(v => (v, Environment.GetEnvironmentVariable(v))).ToArray();
        string root = Path.Combine(Path.GetTempPath(), $"idrak-app-cache-{Environment.ProcessId}");
        string fromVariable = Path.Combine(root, "variable"), set = Path.Combine(root, "set");
        try
        {
            foreach (string name in variables)
            {
                Environment.SetEnvironmentVariable(name, null);
            }

            ComputeResources.CacheFolder = null;
            CpuTuning.UseDefaultCacheFile();
            VulkanBackend.UseDefaultTuningCacheFile();

            // The default: as before where the profile folder is absolute; never relative where it is not.
            string profile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile, Environment.SpecialFolderOption.DoNotVerify);
            string expected = Path.IsPathFullyQualified(profile) ? Path.Combine(profile, ".cache", "idrak") : ComputeResources.CacheFolderFrom(null);
            Check(ComputeResources.CacheFolder == expected && Path.IsPathFullyQualified(ComputeResources.CacheFolder),
                $"unset, the cache folder is .cache/idrak in the profile folder: {ComputeResources.CacheFolder} (expected {expected})");
            string local = Path.Combine(root, "local"), temporary = Path.Combine(root, "temporary");
            Check(ComputeResources.CacheFolderFrom(null, profile: "", localData: local) == Path.Combine(local, "idrak"),
                "an empty profile folder: idrak in the local application data folder");
            Check(ComputeResources.CacheFolderFrom(null, profile: "relative-home", localData: "", temporary: temporary) == Path.Combine(temporary, "idrak"),
                "a relative profile folder and no local application data: idrak in the temporary folder");
            Check(Path.IsPathFullyQualified(ComputeResources.CacheFolderFrom(null, profile: "", localData: "", temporary: "relative-temp")),
                "a relative temporary folder is made absolute");
            Check(ComputeResources.CacheFolderFrom(null, profile: Path.Combine(root, "home")) == Path.Combine(root, "home", ".cache", "idrak"),
                "an absolute profile folder: .cache/idrak in it, as before");

            // IDRAK_CACHE, read each time it is asked (no value captured when the types started).
            Environment.SetEnvironmentVariable("IDRAK_CACHE", fromVariable);
            Check(ComputeResources.CacheFolder == Path.GetFullPath(fromVariable) && Downloader.DefaultCacheRoot == Path.GetFullPath(fromVariable),
                $"IDRAK_CACHE names the cache folder: {ComputeResources.CacheFolder}, the downloader's {Downloader.DefaultCacheRoot}");
            Environment.SetEnvironmentVariable("IDRAK_CACHE", "relative-cache");
            Check(ComputeResources.CacheFolder == Path.GetFullPath("relative-cache") && Path.IsPathFullyQualified(ComputeResources.CacheFolder),
                $"a relative IDRAK_CACHE is made absolute: {ComputeResources.CacheFolder}");
            Environment.SetEnvironmentVariable("IDRAK_CACHE", fromVariable);

            // The setting wins over the variable, and every cache follows it although its type started long ago.
            ComputeResources.CacheFolder = set;
            Check(ComputeResources.CacheFolder == set, $"the setting wins over IDRAK_CACHE: {ComputeResources.CacheFolder}");
            Check(Downloader.DefaultCacheRoot == set && new Downloader().CacheFolder == Path.Combine(set, "downloads"), $"downloads under the setting: {new Downloader().CacheFolder}");
            Check(CpuTuning.CacheFile == Path.Combine(set, "cpu", "tuning.tsv"), $"the CPU's measured cut-overs under the setting, read lazily: {CpuTuning.CacheFile}");
            Check(VulkanBackend.TuningCacheFile == Path.Combine(set, "vulkan", "tuning.tsv"), $"Vulkan's measured choices under the setting, read lazily: {VulkanBackend.TuningCacheFile}");
            Check(TuningCache.Folder() == Path.Combine(set, "tuning", "cuda"), $"CUDA's measured choices under the setting: {TuningCache.Folder()}");
            Check(HipKernelCache.DefaultFolder() == Path.Combine(set, "hip", "kernels"), $"HIP's compiled kernels under the setting: {HipKernelCache.DefaultFolder()}");
            Check(ClusterCache.Folder == Path.Combine(set, "cluster"), $"the cluster's measurements under the setting: {ClusterCache.Folder}");
            Check(FeatureCaches.DefaultFolder == Path.Combine(set, "features"), $"the feature cache under the setting: {FeatureCaches.DefaultFolder}");
            Check(RnNoisePlugin.DefaultModelPath == Path.Combine(set, "rnnoise", "rnnoise.rnnn"), $"the noise model under the setting: {RnNoisePlugin.DefaultModelPath}");
            Check(Environment.GetEnvironmentVariable("IDRAK_VOICES") is { Length: > 0 } || Idrak.Audio.FolderVoiceStore.DefaultRoot == Path.Combine(set, "voices"),
                $"the voice store under the setting: {Idrak.Audio.FolderVoiceStore.DefaultRoot}");
            string prepared = GgufModel.Prepare(TestData(Path.Combine("gguf", "tiny-llama.gguf")));
            Check(prepared.StartsWith(Path.Combine(set, "gguf") + Path.DirectorySeparatorChar, StringComparison.Ordinal) && Directory.Exists(prepared),
                $"a .gguf prepared with no cache root goes under the setting: {prepared}");
            ComputeResources.CacheFolder = "relative-set";
            Check(ComputeResources.CacheFolder == Path.GetFullPath("relative-set"), $"a relative setting is made absolute: {ComputeResources.CacheFolder}");
            Check(Failure<ArgumentException>(() => ComputeResources.CacheFolder = " ").Message.Length > 0, "a blank cache folder is refused");
            ComputeResources.CacheFolder = set;

            // Each cache's own variable still wins.
            string own = Path.Combine(root, "own");
            Environment.SetEnvironmentVariable("IDRAK_CPU_TUNING_FILE", Path.Combine(own, "cpu.tsv"));
            Check(CpuTuning.CacheFile == Path.Combine(own, "cpu.tsv"), $"IDRAK_CPU_TUNING_FILE wins: {CpuTuning.CacheFile}");
            Environment.SetEnvironmentVariable("IDRAK_VULKAN_TUNING_CACHE", Path.Combine(own, "vulkan.tsv"));
            Check(VulkanBackend.TuningCacheFile == Path.Combine(own, "vulkan.tsv"), $"IDRAK_VULKAN_TUNING_CACHE wins: {VulkanBackend.TuningCacheFile}");
            Environment.SetEnvironmentVariable("IDRAK_VULKAN_TUNING_CACHE", "0");
            Check(VulkanBackend.TuningCacheFile is null, "IDRAK_VULKAN_TUNING_CACHE=0 keeps nothing");
            Environment.SetEnvironmentVariable("IDRAK_HIP_KERNEL_CACHE", Path.Combine(own, "hip"));
            Check(HipKernelCache.DefaultFolder() == Path.Combine(own, "hip"), $"IDRAK_HIP_KERNEL_CACHE wins: {HipKernelCache.DefaultFolder()}");
            Environment.SetEnvironmentVariable("IDRAK_TUNING_CACHE", Path.Combine(own, "cuda"));
            Check(TuningCache.Folder() == Path.Combine(own, "cuda"), $"IDRAK_TUNING_CACHE=<folder> wins for CUDA: {TuningCache.Folder()}");
            Environment.SetEnvironmentVariable("IDRAK_TUNING_CACHE", "0");
            Check(TuningCache.Folder() is null && ClusterCache.Folder is null && CpuTuning.DefaultCacheFile() is null,
                "IDRAK_TUNING_CACHE=0 keeps no CUDA, cluster or CPU measurements whatever the cache folder");

            // Null brings back the variable.
            ComputeResources.CacheFolder = null;
            Check(ComputeResources.CacheFolder == Path.GetFullPath(fromVariable), $"null brings back IDRAK_CACHE: {ComputeResources.CacheFolder}");
        }
        finally
        {
            ComputeResources.CacheFolder = null;
            foreach (var (name, value) in saved)
            {
                Environment.SetEnvironmentVariable(name, value);
            }

            CpuTuning.UseDefaultCacheFile();
            VulkanBackend.UseDefaultTuningCacheFile();
            if (Directory.Exists(root))
            {
                Directory.Delete(root, recursive: true);
            }
        }
    }

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
            string message = VulkanDriver.DeviceLostMessage("vkWaitForFences", "vulkan:0 (a test device)", 3, "matmul_tiled", retryOnHost: false);
            foreach (string part in new[] { "vkWaitForFences", "VK_ERROR_DEVICE_LOST", "vulkan:0 (a test device)", "a batch of 3 commands", "'matmul_tiled'",
                         "IDRAK_RETRY_ON_HOST=vulkan", "IDRAK_VULKAN_WIDTH", "cpu" })
            {
                Check(message.Contains(part, StringComparison.Ordinal), $"the lost device's message names {part}: {message}");
            }

            Check(VulkanDriver.DeviceLostMessage("vkQueueSubmit", "d", 1, "", retryOnHost: true).Contains("The host retry is on", StringComparison.Ordinal),
                "with the host retry on, the message says so");
            using (DeviceException.Handled())
            {
                var lost = Failure<VulkanException>(() => VulkanDriver.Check(VulkanDriver.ErrorDeviceLost, "vkAllocateMemory"));
                Check(lost.Message.Contains("vkAllocateMemory", StringComparison.Ordinal) && lost.Message.Contains("IDRAK_RETRY_ON_HOST", StringComparison.Ordinal),
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
