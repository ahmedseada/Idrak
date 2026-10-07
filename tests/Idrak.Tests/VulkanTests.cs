// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using Idrak;
using Idrak.Gpu.Vulkan;

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
        ("vulkan: thousands of small storages share pages under a low allocation cap (IDRAK_VULKAN_MAX_ALLOCATIONS), intact, freed and reused", VulkanSubAllocation),
        ("vulkan: a SPIR-V kernel dispatch (y = a·alpha + b with push constants), chained dispatches in order", VulkanDispatch),
        ("vulkan: uploads, zeroed allocations and reused blocks are ordered after queued dispatches", VulkanOrdering),
        ("vulkan: operations run as generated kernels (no host fallback) and match the CPU (element-wise, rows narrow and wide, products, decoding)", VulkanKernelsMatchCpu),
        ("vulkan: the sampler kernels draw the CPU's tokens with its statistics (greedy, temperature, top-k with and without slots, top-p, min-p, ties)", VulkanSamplerMatchesCpu),
        ("vulkan: the sampler gives the same tokens and statistics run after run (1,000 runs on a GPU; ties, long runs of equal scores)", VulkanSamplerRepeatable),
        ("vulkan: barriers only between dependent dispatches (chains, independent dispatches, write after read), pushed descriptors and sets; dispatches allocate nothing", VulkanBarriers),
        ("vulkan: decoding steps take no host fallback (decoder with float32, int8, int4, bfloat16 weights; multi-head attention; float32, int8, bfloat16 caches; penalties, top-k, top-p, min-p)", VulkanDecodingWithoutFallbacks),
        ("vulkan: kernels shaped by the device (every workgroup width, products split over k, attention split over positions, tiled products, measured and stored choices) match the CPU and give the same bits run after run", VulkanSplitKernels),
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

    private static void ForEachVulkan(Device device, bool separateOnly, Action<VulkanBackend, string> check, int? maxAllocations = null)
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
            var backend = separateOnly ? VulkanBackend.CreateSeparate(ordinal, preferMapped: true, maxAllocations: maxAllocations) : (VulkanBackend)Device.Get("vulkan", ordinal).Backend;
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

            var staging = VulkanBackend.CreateSeparate(ordinal, preferMapped: false, maxAllocations: maxAllocations, stagingBytes: 4 << 20);   // copies in several chunks
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
            var (type, uuid) = VulkanBackend.DeviceKind(i);
            bool software = type == 4;                                   // VK_PHYSICAL_DEVICE_TYPE_CPU
            bool drivenElsewhere = VulkanProvider.DrivenBy(uuid, DeviceProviders.All.TakeWhile(p => p != provider)) is not null;
            Check(provider!.Listed(i) == !(software || drivenElsewhere) && Device.Available.Contains(d) == provider.Listed(i), $"vulkan:{i} listing");
            Check(!software || provider.Note(i) == "software driver: by name only", $"vulkan:{i}: a CPU-type device is reached by name");
            bool optedIn = Environment.GetEnvironmentVariable("IDRAK_VULKAN_DEFAULT") == "1";   // not the default device otherwise
            Check(provider.DefaultRank(i) == (optedIn ? type switch { 2 => 50, 1 => 40, _ => (int?)null } : null), $"vulkan:{i} rank");
            Check(Device.Parse($"vulkan:{i}") == d && d.Type == DeviceType.Vulkan && d.IsGpu && d.ToString() == $"vulkan:{i}", $"vulkan:{i} by name");
            var backend = (VulkanBackend)d.Backend;
            Check(provider.IsStarted(i) && d.Name.Contains("Vulkan", StringComparison.Ordinal), $"vulkan:{i} started, named");
            Check(!backend.Capabilities.MatrixUnits && backend.Capabilities.FusedKernels && !backend.Capabilities.Profiling, "fused kernels; no matrix units or profiling yet");
            Check(backend.MaxStorageBytes >= 1 << 27, $"maxStorageBufferRange {backend.MaxStorageBytes} (at least 128 MiB)");
            Console.WriteLine($"    vulkan:{i}: {d.Name}; type {type}, uuid {uuid}, {(backend.UnifiedMemory ? "mapped" : "staging")} copies, " +
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

    // Drivers cap memory allocations (4096 on AMD's Windows driver); under a cap of 8, thousands of live storages share pages.
    private static void VulkanSubAllocation(Device device) => ForEachVulkan(device, separateOnly: true, (backend, label) =>
    {
        const int Cap = 8, Count = 3000;
        Check(backend.MaxAllocations == Cap && backend.SubAllocationMax == backend.PageBytes / 4, $"{label}: capped at {Cap} allocations");
        int Length(int i) => i * 37 % 2000 + 1;
        float[] Expected(int i) => [.. Enumerable.Range(0, Length(i)).Select(j => i + j * 1e-4f)];
        var storages = new List<Storage>();
        void AllocateAll()
        {
            for (int i = 0; i < Count; i++)
            {
                var storage = backend.Allocate(Length(i), zeroed: i % 3 == 0);
                backend.Upload(Expected(i), storage);
                storages.Add(storage);
            }
        }

        void CheckAll(string what)
        {
            for (int i = 0; i < Count; i++)
            {
                var values = new float[Length(i)];
                backend.Download(storages[i], values);
                Check(values.AsSpan().SequenceEqual(Expected(i)), $"{label}: storage {i} intact ({what})");
            }
        }

        try
        {
            AllocateAll();
            Check(backend.Allocations <= Cap && backend.PageCount >= 1, $"{label}: {Count} storages in {backend.PageCount} pages, {backend.Allocations} allocations");

            // Kernels bind carved storages like any other (y[0] = 2 · storage 0 [0] + storage 2000 [0]).
            QueueAxpb(backend, storages[0], storages[2000], storages[1000], 1, 2f);
            var one = new float[1];
            backend.Download(storages[1000], one);
            Check(one[0] == 2000f, $"{label}: a dispatch over carved storages ({one[0]})");
            backend.Upload(Expected(1000), storages[1000]);

            // A large storage gets an allocation of its own.
            int before = backend.Allocations;
            var big = backend.Allocate((int)(backend.SubAllocationMax / 4) + 1, zeroed: true);
            Check(backend.Allocations == before + 1, $"{label}: a storage over a quarter page has its own allocation");
            CheckAll("beside a large storage");
            big.Release();

            // Past the cap: the existing error, after freeing what is cached.
            var bigs = new List<Storage>();
            try
            {
                for (int i = 0; i <= Cap; i++)
                {
                    bigs.Add(backend.Allocate((int)(backend.SubAllocationMax / 4) + 1, zeroed: false));
                }

                throw new Exception($"{label}: allocations past the cap succeeded");
            }
            catch (ResourceLimitExceededException ex)
            {
                Check(ex.Message.Contains("memory allocations", StringComparison.Ordinal), $"{label}: {ex.Message}");
            }
            finally
            {
                bigs.ForEach(b => b.Release());
            }

            CheckAll("after the cap was reached");

            // Freed, pages given back, then everything again (reusing pooled blocks and fresh pages).
            storages.ForEach(s => s.Release());
            storages.Clear();
            backend.ReleaseCachedMemory();
            Check(backend.PageCount == 0 && backend.GetMemoryUsage().InUse == 0, $"{label}: empty pages freed ({backend.PageCount} left, {backend.GetMemoryUsage()})");
            AllocateAll();
            for (int i = 0; i < Count; i += 2)
            {
                storages[i].Release();                                       // every other one freed and reused
                storages[i] = backend.Allocate(Length(i), zeroed: false);
                backend.Upload(Expected(i), storages[i]);
            }

            CheckAll("freed and reused");
            Check(backend.Allocations <= Cap, $"{label}: still within the cap ({backend.Allocations})");
        }
        finally
        {
            storages.ForEach(s => s.Release());
        }
    }, maxAllocations: 8);

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
        const int N = 4096;
        int Steps = 2 * backend.MaxBatchCommands + 88;                 // the batch size is measured on the device
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

    private static Storage CpuStorage(int n) => Idrak.Abstraction.Devices.Cpu.CpuBackend.Instance.Allocate(n, zeroed: true);

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
            using (var host = new HostCall(backend))
            {
                Idrak.Abstraction.Devices.Cpu.CpuBackend.Instance.Copy(host[y], host[a], N);  // a = y, through the host
            }

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

    // Each case runs one operation on the CPU and on the Vulkan backend from the same inputs; every storage of the case is
    // compared after it. On the Vulkan backend it must dispatch kernels and never take the host fallback.
    private static void VulkanKernelsMatchCpu(Device device) => ForEachVulkan(device, (backend, label) =>
    {
        if (Environment.GetEnvironmentVariable("IDRAK_VULKAN_KERNELS") is "0" or "false")
        {
            Console.WriteLine("    (IDRAK_VULKAN_KERNELS=0: kernels off, skipped)");
            return;
        }

        var cpu = Idrak.Abstraction.Devices.Cpu.CpuBackend.Instance;
        var random = new Random(9);
        float[] R(int n, float lo = -2f, float hi = 2f) => [.. Enumerable.Range(0, n).Select(_ => lo + (hi - lo) * random.NextSingle())];
        float[] Bits(int n) => [.. Enumerable.Range(0, n).Select(_ => BitConverter.Int32BitsToSingle((random.Next() ^ (random.Next() << 16)) & ~(1 << 30)))];   // random bytes, never NaN
        float[] Ints(int n, int max) => [.. Enumerable.Range(0, n).Select(_ => (float)random.Next(max))];

        void Case(string what, float[][] inputs, Action<Backend, Storage[]> op, float tolerance = 2e-5f)
        {
            var host = inputs.Select(d => { var st = cpu.Allocate(d.Length, false); cpu.Upload(d, st); return st; }).ToArray();
            var gpu = inputs.Select(d => { var st = backend.Allocate(d.Length, false); backend.Upload(d, st); return st; }).ToArray();
            try
            {
                long dispatches = backend.Dispatches, fallbacks = Idrak.Abstraction.Operations.Kernels.HostCalls(backend);
                op(cpu, host);
                op(backend, gpu);
                Check(backend.Dispatches > dispatches, $"{label}: {what} dispatched no kernel");
                Check(Idrak.Abstraction.Operations.Kernels.HostCalls(backend) == fallbacks, $"{label}: {what} took the host fallback");
                for (int i = 0; i < inputs.Length; i++)
                {
                    var (expected, actual) = (new float[inputs[i].Length], new float[inputs[i].Length]);
                    cpu.Download(host[i], expected);
                    backend.Download(gpu[i], actual);
                    AssertClose(expected, actual, tolerance, $"{label}: {what}, storage {i}");
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

        const int N = 70_001;
        Case("unary gelu", [R(N), new float[N]], (b, s) => b.Unary(UnaryOp.Gelu, s[0], s[1], N));
        Case("unary backward tanh", [R(N), R(N, -1, 1), R(N), R(N)], (b, s) => b.UnaryBackward(UnaryOp.Tanh, s[0], s[1], s[2], s[3], N));
        Case("binary mul, axpy", [R(N), R(N), new float[N]], (b, s) =>
        {
            b.Binary(BinaryOp.Mul, s[0], s[1], s[2], N);
            b.Axpy(s[0], s[2], N, 0.5f);
        });
        Case("add row vector, sum rows", [R(37 * 301), R(301), new float[37 * 301], R(301)], (b, s) =>
        {
            b.AddRowVector(s[0], s[1], s[2], 37, 301);
            b.SumRows(s[2], s[3], 37, 301);
        });
        Case("sum", [R(N), new float[1]], (b, s) => b.Sum(s[0], s[1], N, 0.5f), 1e-4f);
        foreach (int cols in new[] { 33, 700 })                              // narrow (one invocation per row) and wide rows
        {
            int rows = 40;
            Case($"softmax and log-softmax, {cols} columns", [R(rows * cols, -5, 5), new float[rows * cols], new float[rows * cols]], (b, s) =>
            {
                b.Softmax(s[0], s[1], rows, cols, log: false);
                b.Softmax(s[0], s[2], rows, cols, log: true);
            });
            Case($"masked softmax, {cols} columns", [R(rows * cols), R(4 * cols), new float[rows * cols]], (b, s) => b.ScaleMaskSoftmax(s[0], s[1], s[2], rows, cols, 4, 0.3f));
            Case($"rms norm with gain, layer norm, {cols} columns", [R(rows * cols), R(cols), R(cols), new float[rows * cols], new float[rows * cols]], (b, s) =>
            {
                b.RmsNormAffine(s[0], s[1], s[3], rows, cols, 1e-6f, 1f);
                b.LayerNormFused(s[0], s[1], s[2], s[4], rows, cols, 1e-5f);
            });
            Case($"arg max, {cols} columns", [R(rows * cols), new float[rows]], (b, s) => b.ArgMax(s[0], s[1], rows, cols), 0f);
        }

        foreach (var (batch, m, n, k) in new[] { (3, 5, 7, 9), (1, 70, 65, 80) })   // the small and the tiled product
        {
            foreach (bool transB in new[] { false, true })
            {
                Case($"batched matmul {batch}×{m}×{n}×{k}, transB {transB}", [R(batch * m * k), R(batch * k * n), R(batch * m * n)],
                    (b, s) => b.BatchedMatMul(s[0], s[1], s[2], batch, m, n, k, false, transB, 0.5f));
            }
        }

        Case("int8 product", [R(3 * 64), Bits(64 * 18), R(70), new float[3 * 70]], (b, s) => b.Int8MatMul(s[0], s[1], s[2], s[3], 3, 70, 64), 5e-4f);   // products of up to ±127 cancel: GPUs fuse multiply-adds
        Case("bfloat16 product, packing", [R(3 * 64), R(64 * 70), new float[64 * 35], new float[3 * 70]], (b, s) =>
        {
            b.PackBFloat16(s[1], s[2], 64 * 70);
            b.BFloat16MatMul(s[0], s[2], s[3], 3, 70, 64);
        });
        Case("rope, gather, permute, copy 2d", [R(24 * 12), R(24 * 12), R(20 * 5), R(20 * 5), Ints(3, 20), R(50 * 12), Ints(9, 50), new float[9 * 12], new float[9 * 12], R(60)],
            (b, s) =>
            {
                b.Rope(s[0], s[1], s[2], s[3], s[4], 24, 4, 3, 12, 5, false, 1f);
                b.Gather(s[5], s[6], s[7], 9, 12, 50);
                b.Permute(s[7], s[8], [12, 9], [1, 12], accumulate: false);
                b.Copy2D(s[8], 2, 12, s[9], 1, 10, 5, 7, accumulate: true);
            });
        foreach (int dim in new[] { 48, 128 })                              // the 64-wide and the 256-wide attention
        {
            int heads = 2, rowsPerHead = 3, capacity = 40;
            Case($"key/value write and decoding attention, head size {dim}", [R(heads * rowsPerHead * dim), R(heads * 2 * dim), R(heads * capacity * dim), R(heads * capacity * dim), new float[] { 20f }, new float[heads * rowsPerHead * dim]],
                (b, s) =>
                {
                    b.KeyValueWrite(s[1], s[2], s[4], heads, 2, capacity, dim);
                    b.AttentionDecode(s[0], s[2], s[3], s[4], s[5], heads, rowsPerHead, 3, capacity, dim, 0.125f);
                });
        }

        Case("dropout", [R(N), new float[N]], (b, s) => b.Dropout(s[0], s[1], N, 0.25f, 77u));

        // Decoding steps: normalized and rotated queries and keys (narrow and wide rows, partial and interleaved rotation),
        // repetition penalties over a wrapped history ring, and the history itself.
        foreach (var (cols, half, interleaved) in new[] { (12, 4, false), (12, 6, true), (128, 64, false), (128, 32, true) })
        {
            int steps = 3, heads = 4, heads2 = 2, rows1 = steps * heads, rows2 = steps * heads2;
            Case($"rms norm + rope, {cols} columns, half {half}, interleaved {interleaved}",
                [R(rows1 * cols), R(cols), new float[rows1 * cols], R(rows2 * cols), R(cols), new float[rows2 * cols], R(20 * half), R(20 * half), Ints(steps, 20), new float[rows1 * cols]],
                (b, s) =>
                {
                    b.RmsNormRopePair(s[0], s[1], s[2], rows1, 1e-6f, 1f, heads, s[3], s[4], s[5], rows2, 1e-5f, 0f, heads2, s[6], s[7], s[8], cols, steps, half, interleaved);
                    b.RmsNormRope(s[0], s[1], s[6], s[7], s[8], s[9], rows1, cols, 1e-6f, 0.5f, heads, steps, half, interleaved);
                }, 1e-4f);
        }

        int vocabulary = 300, ring = 16;
        float[] history = [.. Ints(2 * ring, 12).Select((v, i) => i % 7 == 3 ? 500f : i % 11 == 5 ? -2f : v)];   // repeats, out-of-range ids
        foreach (float length in new[] { 0f, 5f, 25f })
        {
            Case($"penalties and history, {length} tokens seen", [R(2 * 3 * vocabulary), new float[2 * vocabulary], history, [length], [7f, 299f]], (b, s) =>
            {
                b.PenalizeRows(s[0], s[1], s[2], s[3], 2, vocabulary, 3 * vocabulary, 2 * vocabulary, ring, 10, 1.3f, 0.2f, 0.05f);
                b.HistoryPush(s[4], s[2], s[3], 2, ring);
            }, 1e-6f);                                                   // x / repeat: GPUs divide to 2.5 ULP (Vulkan), the CPU exactly
        }
    });
}
