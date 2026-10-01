// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using Idrak;
using Idrak.Backends;
using Idrak.Backends.Vulkan;

// The Vulkan runtime: devices, memory, copies and SPIR-V dispatches. Run on a Vulkan device, the checks test that device;
// run on the CPU, they test every Vulkan device found, listed or not (lavapipe, Mesa's software driver, here). Each runs
// twice per device: as the device copies (mapped memory on integrated GPUs and CPU drivers) and through a staging
// buffer. Without Vulkan they pass with a note.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] VulkanGroup =
    [
        ("vulkan: devices are found (or the reason why not), listed unless a software driver or a GPU CUDA drives", VulkanDevices),
        ("vulkan: uploads and downloads round-trip (1, odd and multi-megabyte sizes, ranges), mapped and through staging", VulkanRoundTrip),
        ("vulkan: memory is pooled, counted and limited by ComputeResources.GpuMemoryLimit", VulkanMemory),
        ("vulkan: a SPIR-V kernel dispatch (y = a·alpha + b with push constants), chained dispatches in order", VulkanDispatch),
        ("vulkan: uploads, zeroed allocations and reused blocks are ordered after queued dispatches", VulkanOrdering),
    ];

    // y[i] = a[i] * alpha + b[i] for i < n; bindings a, b, y; push constants { uint n; float alpha; }; 64 lanes per group.
    // Assembled with `spirv-as --target-env vulkan1.1` (SPIR-V 1.3) from:
    //   OpCapability Shader / OpMemoryModel Logical GLSL450 / OpEntryPoint GLCompute %main "main" %gid
    //   OpExecutionMode %main LocalSize 64 1 1; %gid = GlobalInvocationId; %Buf = Block { float[] } (ArrayStride 4);
    //   %a, %b, %y: StorageBuffer %Buf at set 0, bindings 0, 1, 2; %pc: PushConstant Block { uint n (0); float alpha (4) }
    //   main: i = gid.x; if (i < n) y[i] = a[i] * alpha + b[i]
    private static readonly uint[] AxpbSpirv =
    [
        0x07230203, 0x00010300, 0x00070000, 0x0000002C, 0x00000000, 0x00020011, 0x00000001, 0x0003000E,
        0x00000000, 0x00000001, 0x0006000F, 0x00000005, 0x00000001, 0x6E69616D, 0x00000000, 0x00000002,
        0x00060010, 0x00000001, 0x00000011, 0x00000040, 0x00000001, 0x00000001, 0x00040047, 0x00000002,
        0x0000000B, 0x0000001C, 0x00040047, 0x00000003, 0x00000006, 0x00000004, 0x00050048, 0x00000004,
        0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x00000004, 0x00000002, 0x00040047, 0x00000005,
        0x00000022, 0x00000000, 0x00040047, 0x00000005, 0x00000021, 0x00000000, 0x00040047, 0x00000006,
        0x00000022, 0x00000000, 0x00040047, 0x00000006, 0x00000021, 0x00000001, 0x00040047, 0x00000007,
        0x00000022, 0x00000000, 0x00040047, 0x00000007, 0x00000021, 0x00000002, 0x00050048, 0x00000008,
        0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x00000008, 0x00000001, 0x00000023, 0x00000004,
        0x00030047, 0x00000008, 0x00000002, 0x00020013, 0x00000009, 0x00030021, 0x0000000A, 0x00000009,
        0x00040015, 0x0000000B, 0x00000020, 0x00000000, 0x00040015, 0x0000000C, 0x00000020, 0x00000001,
        0x00030016, 0x0000000D, 0x00000020, 0x00020014, 0x0000000E, 0x00040017, 0x0000000F, 0x0000000B,
        0x00000003, 0x00040020, 0x00000010, 0x00000001, 0x0000000F, 0x00040020, 0x00000011, 0x00000001,
        0x0000000B, 0x0004003B, 0x00000010, 0x00000002, 0x00000001, 0x0003001D, 0x00000003, 0x0000000D,
        0x0003001E, 0x00000004, 0x00000003, 0x00040020, 0x00000012, 0x0000000C, 0x00000004, 0x0004003B,
        0x00000012, 0x00000005, 0x0000000C, 0x0004003B, 0x00000012, 0x00000006, 0x0000000C, 0x0004003B,
        0x00000012, 0x00000007, 0x0000000C, 0x0004001E, 0x00000008, 0x0000000B, 0x0000000D, 0x00040020,
        0x00000013, 0x00000009, 0x00000008, 0x0004003B, 0x00000013, 0x00000014, 0x00000009, 0x00040020,
        0x00000015, 0x00000009, 0x0000000B, 0x00040020, 0x00000016, 0x00000009, 0x0000000D, 0x00040020,
        0x00000017, 0x0000000C, 0x0000000D, 0x0004002B, 0x0000000C, 0x00000018, 0x00000000, 0x0004002B,
        0x0000000C, 0x00000019, 0x00000001, 0x0004002B, 0x0000000B, 0x0000001A, 0x00000000, 0x00050036,
        0x00000009, 0x00000001, 0x00000000, 0x0000000A, 0x000200F8, 0x0000001B, 0x00050041, 0x00000011,
        0x0000001C, 0x00000002, 0x0000001A, 0x0004003D, 0x0000000B, 0x0000001D, 0x0000001C, 0x00050041,
        0x00000015, 0x0000001E, 0x00000014, 0x00000018, 0x0004003D, 0x0000000B, 0x0000001F, 0x0000001E,
        0x000500B0, 0x0000000E, 0x00000020, 0x0000001D, 0x0000001F, 0x000300F7, 0x00000021, 0x00000000,
        0x000400FA, 0x00000020, 0x00000022, 0x00000021, 0x000200F8, 0x00000022, 0x00050041, 0x00000016,
        0x00000023, 0x00000014, 0x00000019, 0x0004003D, 0x0000000D, 0x00000024, 0x00000023, 0x00060041,
        0x00000017, 0x00000025, 0x00000005, 0x00000018, 0x0000001D, 0x0004003D, 0x0000000D, 0x00000026,
        0x00000025, 0x00060041, 0x00000017, 0x00000027, 0x00000006, 0x00000018, 0x0000001D, 0x0004003D,
        0x0000000D, 0x00000028, 0x00000027, 0x00050085, 0x0000000D, 0x00000029, 0x00000026, 0x00000024,
        0x00050081, 0x0000000D, 0x0000002A, 0x00000029, 0x00000028, 0x00060041, 0x00000017, 0x0000002B,
        0x00000007, 0x00000018, 0x0000001D, 0x0003003E, 0x0000002B, 0x0000002A, 0x000200F9, 0x00000021,
        0x000200F8, 0x00000021, 0x000100FD, 0x00010038,
    ];

    private static readonly VulkanKernel Axpb = new(AxpbSpirv, bindings: 3, pushConstantBytes: 8, name: "axpb");

    [StructLayout(LayoutKind.Sequential)]
    private struct AxpbArguments
    {
        public uint N;
        public float Alpha;
    }

    // Queues y = a * alpha + b over n elements.
    private static void QueueAxpb(VulkanBackend backend, Storage a, Storage b, Storage y, int n, float alpha)
    {
        var arguments = new AxpbArguments { N = (uint)n, Alpha = alpha };
        backend.Dispatch(Axpb, (uint)((n + 63) / 64), 1, 1, [a, b, y], MemoryMarshal.AsBytes(new ReadOnlySpan<AxpbArguments>(in arguments)));
    }

    // The Vulkan backends to test from `device` (none on a CUDA device: those run from the CPU's pass); each is given
    // to `check` as the device copies and then through staging, on a second backend of the same device.
    private static void ForEachVulkan(Device device, Action<VulkanBackend, string> check) => ForEachVulkan(device, separateOnly: false, check);

    private static void ForEachVulkan(Device device, bool separateOnly, Action<VulkanBackend, string> check)
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

        foreach (int ordinal in ordinals)
        {
            var backend = separateOnly ? VulkanBackend.CreateSeparate(ordinal, preferMapped: true) : (VulkanBackend)Device.Get("vulkan", ordinal).Backend;
            try
            {
                check(backend, $"vulkan:{ordinal} ({(backend.UnifiedMemory ? "mapped" : "staging")})");
            }
            finally
            {
                if (separateOnly)
                {
                    backend.Shutdown();
                }
            }

            var staging = VulkanBackend.CreateSeparate(ordinal, preferMapped: false);
            try
            {
                Check(!staging.UnifiedMemory, "the second backend copies through staging");
                check(staging, $"vulkan:{ordinal} (staging)");
            }
            finally
            {
                staging.Shutdown();
            }
        }
    }

    private static float[] Values(int n, int seed)
    {
        var random = new Random(seed);
        return [.. Enumerable.Range(0, n).Select(_ => random.NextSingle() * 2 - 1)];
    }

    private static void VulkanDevices(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;                                                      // checked once
        }

        var provider = DeviceProviders.Find("vulkan");
        Check(provider is not null && provider.Type == DeviceType.Vulkan && provider.Display == "Vulkan", "the Vulkan provider is registered");
        Check(DeviceProviders.All[0].Kind == "cuda" && DeviceProviders.All[1].Kind == "vulkan", "registered after CUDA");
        if (VulkanBackend.DeviceCount == 0)
        {
            Check(VulkanBackend.UnavailableReason.Length > 0, "a reason when there is no device");
            Console.WriteLine($"    (no Vulkan device: {VulkanBackend.UnavailableReason})");
            return;
        }

        for (int i = 0; i < VulkanBackend.DeviceCount; i++)
        {
            var d = Device.Get("vulkan", i);
            var (type, vendor) = VulkanBackend.DeviceKind(i);
            bool software = type == 4;                                   // VK_PHYSICAL_DEVICE_TYPE_CPU
            bool cudaDrives = vendor == 0x10DE && Device.IsCudaAvailable;
            Check(provider!.Listed(i) == !(software || cudaDrives) && Device.Available.Contains(d) == provider.Listed(i), $"vulkan:{i} listing");
            Check(provider.DefaultRank(i) == type switch { 2 => 50, 1 => 40, _ => (int?)null }, $"vulkan:{i} rank");
            Check(Device.Parse($"vulkan:{i}") == d && d.Type == DeviceType.Vulkan && d.IsGpu && d.ToString() == $"vulkan:{i}", $"vulkan:{i} by name");
            var backend = (VulkanBackend)d.Backend;
            Check(provider.IsStarted(i) && d.Name.Contains("Vulkan", StringComparison.Ordinal), $"vulkan:{i} started, named");
            Check(!backend.Capabilities.MatrixUnits && !backend.Capabilities.FusedKernels && !backend.Capabilities.Profiling, "no matrix units, fused kernels or profiling yet");
            Check(backend.MaxStorageBytes >= 1 << 27, $"maxStorageBufferRange {backend.MaxStorageBytes} (at least 128 MiB)");
            Console.WriteLine($"    vulkan:{i}: {d.Name}; type {type}, vendor 0x{vendor:X4}, {(backend.UnifiedMemory ? "mapped" : "staging")} copies, " +
                $"storages up to {backend.MaxStorageBytes >> 20} MiB{(provider.Listed(i) ? "" : ", not listed")}");
        }
    }

    private static void VulkanRoundTrip(Device device) => ForEachVulkan(device, (backend, label) =>
    {
        foreach (int n in new[] { 1, 3, 1001, 65_537, 3_000_001, 5_000_000 })              // up to 20 MB: several staging chunks
        {
            var values = Values(n, n);
            var storage = backend.Allocate(n, zeroed: false);
            try
            {
                backend.Upload(values, storage);
                var back = new float[n];
                backend.Download(storage, back);
                Check(back.AsSpan().SequenceEqual(values), $"{label}: {n} floats round-trip");
                int offset = n / 3, length = n - offset - n / 5;
                var range = new float[length];
                backend.DownloadRange(storage, offset, range);
                Check(range.AsSpan().SequenceEqual(values.AsSpan(offset, length)), $"{label}: {n} floats, a range from {offset}");
                backend.Upload(values.AsSpan(0, n / 2), storage);                                // a prefix leaves the rest
                backend.Download(storage, back);
                Check(back.AsSpan().SequenceEqual(values), $"{label}: {n} floats, prefix upload");
            }
            finally
            {
                storage.Release();
            }

            var zeros = backend.Allocate(n, zeroed: true);
            var read = new float[n];
            Array.Fill(read, 1f);
            backend.Download(zeros, read);
            zeros.Release();
            Check(read.All(v => v == 0f), $"{label}: {n} zeroed floats");
        }

        // A tensor on the device itself (for the device's own backend).
        for (int i = 0; i < VulkanBackend.DeviceCount; i++)
        {
            if (VulkanBackend.IsInitialized(i) && ReferenceEquals(VulkanBackend.Get(i), backend))
            {
                var x = Values(777, 5);
                using var t = Tensor.From(x, [7, 111], Device.Get("vulkan", i));
                Check(t.ToArray().AsSpan().SequenceEqual(x), $"{label}: a tensor round-trips");
            }
        }
    });

    // On backends of their own: the device's backend counts the other tests' tensors as they are finalized.
    private static void VulkanMemory(Device device) => ForEachVulkan(device, separateOnly: true, (backend, label) =>
    {
        var before = backend.GetMemoryUsage();
        var a = backend.Allocate(1000, zeroed: false);
        Check(backend.GetMemoryUsage().InUse == before.InUse + 4000, $"{label}: 4000 bytes in use");
        a.Release();
        var after = backend.GetMemoryUsage();
        Check(after.InUse == before.InUse && after.Cached >= 4000, $"{label}: released to the pool ({after})");
        var b = backend.Allocate(1000, zeroed: false);
        Check(backend.GetMemoryUsage().Cached == after.Cached - 4000, $"{label}: the pooled block is reused");
        b.Release();
        backend.ReleaseCachedMemory();
        Check(backend.GetMemoryUsage().Cached == 0, $"{label}: the cache is freed");

        long? old = ComputeResources.GpuMemoryLimit;
        try
        {
            ComputeResources.GpuMemoryLimit = backend.GetMemoryUsage().InUse + 1_000_000;
            var small = backend.Allocate(1000, zeroed: true);
            small.Release();
            try
            {
                var big = backend.Allocate(1_000_000, zeroed: false);
                big.Release();
                throw new Exception($"{label}: an allocation over the limit succeeded");
            }
            catch (ResourceLimitExceededException)
            {
            }
        }
        finally
        {
            ComputeResources.GpuMemoryLimit = old;
        }

        // Eviction gives the memory back; restoring gives fresh memory and recomputes the values.
        var evicted = backend.Allocate(64, zeroed: false);
        backend.Upload(Values(64, 1), evicted);
        backend.Evict(evicted, s => backend.Upload(Values(64, 1), s));
        Check(evicted.Evicted && backend.Restore(evicted), $"{label}: evicted and restored");
        var restored = new float[64];
        backend.Download(evicted, restored);
        evicted.Release();
        Check(restored.AsSpan().SequenceEqual(Values(64, 1)), $"{label}: restored values");
    });

    private static void VulkanDispatch(Device device) => ForEachVulkan(device, (backend, label) =>
    {
        foreach (int n in new[] { 1, 63, 64, 1000, 100_003 })
        {
            var (av, bv) = (Values(n, 1), Values(n, 2));
            var a = backend.Allocate(n, zeroed: false);
            var b = backend.Allocate(n, zeroed: false);
            var y = backend.Allocate(n + 1, zeroed: false);
            var z = backend.Allocate(n + 1, zeroed: false);
            try
            {
                backend.Upload(av, a);
                backend.Upload(bv, b);
                backend.Upload(new float[n + 1].Select(_ => 7f).ToArray(), y);
                long dispatches = backend.Dispatches;
                QueueAxpb(backend, a, b, y, n, 2.5f);                      // y = 2.5a + b (y[n] untouched)
                QueueAxpb(backend, y, a, z, n, -1f);                       // z = a - y = -1.5a - b: reads the first's output
                Check(backend.Dispatches == dispatches + 2, $"{label}: two dispatches queued");
                var result = new float[n + 1];
                backend.Download(y, result);
                AssertClose([.. av.Zip(bv, (x, w) => x * 2.5f + w), 7f], result, 1e-6f, $"{label}: y = 2.5a + b over {n}");
                backend.Download(z, result.AsSpan(0, n));
                AssertClose([.. av.Zip(bv, (x, w) => -1.5f * x - w)], result[..n], 1e-5f, $"{label}: chained dispatch over {n}");
            }
            finally
            {
                a.Release();
                b.Release();
                y.Release();
                z.Release();
            }
        }

        // A long chain spans several batches (submitted on their own), still in order.
        const int N = 4096, Steps = 600;
        var x = backend.Allocate(N, zeroed: true);
        var one = backend.Allocate(N, zeroed: false);
        backend.Upload(Enumerable.Repeat(1f, N).ToArray(), one);
        long submissions = backend.Submissions;
        for (int step = 0; step < Steps; step++)
        {
            QueueAxpb(backend, x, one, x, N, 1f);                          // x += 1, in place
        }

        var sum = new float[N];
        backend.Download(x, sum);
        x.Release();
        one.Release();
        Check(sum.All(v => v == Steps), $"{label}: {Steps} in-place dispatches give {Steps} (got {sum[0]}, {sum[^1]})");
        Check(backend.Submissions - submissions >= 2, $"{label}: long chains submitted in several batches");

        // Wrong arguments are refused before anything is queued.
        var s = backend.Allocate(4, zeroed: true);
        try
        {
            Check(Throws<ArgumentException>(() => backend.Dispatch(Axpb, 1, 1, 1, [s, s], new byte[8])), "too few storages");
            Check(Throws<ArgumentException>(() => backend.Dispatch(Axpb, 1, 1, 1, [s, s, s], new byte[4])), "wrong push-constant size");
            var cpu = CpuStorage(4);
            Check(Throws<ArgumentException>(() => backend.Dispatch(Axpb, 1, 1, 1, [s, s, cpu], new byte[8])), "a storage of another device");
            cpu.Release();
        }
        finally
        {
            s.Release();
        }
    });

    private static Storage CpuStorage(int n) => Idrak.Backends.Cpu.CpuBackend.Instance.Allocate(n, zeroed: true);

    private static bool Throws<T>(Action action) where T : Exception
    {
        try
        {
            action();
            return false;
        }
        catch (T)
        {
            return true;
        }
    }

    private static void VulkanOrdering(Device device) => ForEachVulkan(device, (backend, label) =>
    {
        const int N = 50_000;
        var (av, bv) = (Values(N, 3), Values(N, 4));
        var a = backend.Allocate(N, zeroed: false);
        var b = backend.Allocate(N, zeroed: false);
        var y = backend.Allocate(N, zeroed: false);
        try
        {
            // An upload after a queued dispatch must not change what the dispatch reads.
            backend.Upload(av, a);
            backend.Upload(bv, b);
            QueueAxpb(backend, a, b, y, N, 3f);
            backend.Upload(new float[N], a);                               // a = 0 after the dispatch
            QueueAxpb(backend, a, b, b, N, 1f);                            // b = 0 + b: still bv
            var result = new float[N];
            backend.Download(y, result);
            AssertClose([.. av.Zip(bv, (x, w) => 3f * x + w)], result, 1e-6f, $"{label}: the dispatch read a before the upload");
            backend.Download(b, result);
            Check(result.AsSpan().SequenceEqual(bv), $"{label}: the second dispatch read the uploaded zeros");

            // The host fallback (download, CPU kernel, upload) between dispatches.
            QueueAxpb(backend, b, b, y, N, 1f);                            // y = 2b
            backend.Copy(y, a, N);                                         // a = y, through the host
            QueueAxpb(backend, a, b, y, N, 1f);                            // y = 3b
            backend.Download(y, result);
            AssertClose([.. bv.Select(v => 3f * v)], result, 1e-6f, $"{label}: dispatch, host fallback, dispatch");

            // A block released while a dispatch still uses it, reused at once (zeroed): the zeros come after the dispatch.
            var t = backend.Allocate(N, zeroed: false);
            backend.Upload(av, t);
            QueueAxpb(backend, t, b, y, N, 1f);                            // y = a + b, reading t
            t.Release();
            var reused = backend.Allocate(N, zeroed: true);                // t's block from the pool
            backend.Download(y, result);
            AssertClose([.. av.Zip(bv, (x, w) => x + w)], result, 1e-6f, $"{label}: the dispatch read the block before it was zeroed");
            backend.Download(reused, result);
            reused.Release();
            Check(result.All(v => v == 0f), $"{label}: the reused block is zeroed");
            backend.Synchronize();
        }
        finally
        {
            a.Release();
            b.Release();
            y.Release();
        }
    });
}
