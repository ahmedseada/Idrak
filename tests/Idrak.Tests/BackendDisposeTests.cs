// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak;
using Idrak.Abstraction;
using Idrak.Abstraction.Cluster;
using Idrak.Abstraction.Devices;
using Idrak.Cluster;
using Idrak.Gpu.Vulkan;

// A separate backend is disposable (plan 36, G-1b, item G6): its memory, kernels and streams go back to the device, a
// call after that fails naming it, and the device's own backend lives for the process. On a GPU, creating and disposing a
// separate backend again and again leaves the device's free memory where it started (rule 77).
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] BackendDisposeGroup =
    [
        ("backend dispose: a separate CPU backend gives its pool back; Allocate, Upload, Download and an operation after it throw ObjectDisposedException naming the backend; Return is a no-op and a second Dispose does nothing; the device's own backend ignores Dispose", d => { CpuOnly(d); SeparateCpuBackendDisposes(); }),
        ("backend dispose: a miss in the CPU pool while it caches more than is in use frees only as much as it needs, the least recently used sizes first (rule 77: in use plus cached does not grow past twice what is in use), so the sizes still in use stay cached and hit; while it caches no more than is in use, a miss frees nothing and every cached size hits again (a cycle of sizes, such as a prompt pass's layers between decoding steps, keeps its blocks)", d => { CpuOnly(d); CpuPoolFreesOnlyWhatAMissNeeds(); }),
        ("backend dispose: creating, using and disposing a separate backend of the GPU again and again leaves the driver's free memory where it started (within its granularity; not measured where the driver reports a CPU device, whose free memory is the host's), and each disposed one refuses calls", SeparateGpuBackendReturnsMemory),
    ];

    private static void SeparateCpuBackendDisposes()
    {
        var backend = Device.Cpu.CreateSeparateBackend(new SeparateBackendOptions("backend dispose test", 16L << 20));
        Check(!backend.IsDisposed, "a new separate backend is not disposed");
        var kept = backend.Allocate(256, zeroed: true);
        var pooled = backend.Allocate(512, zeroed: false);
        backend.Fill(kept, 256, 2f);
        float[] read = new float[256];
        backend.Download(kept, read);
        Check(read.All(v => v == 2f), "the separate backend computes before it is disposed");
        backend.Return(pooled);
        Check(backend.GetMemoryUsage().Cached > 0, $"a returned block is cached in the backend's pool: {backend.GetMemoryUsage()}");

        backend.Dispose();
        Check(backend.IsDisposed, "Dispose marks the backend disposed");
        Check(backend.GetMemoryUsage().Cached == 0, $"Dispose gives the pool's cached blocks back: {backend.GetMemoryUsage()}");
        ThrowsNaming<ObjectDisposedException>(() => backend.Allocate(16, zeroed: false), "Allocate after Dispose", backend.Name, "disposed");
        ThrowsNaming<ObjectDisposedException>(() => backend.Upload([1f, 2f], kept), "Upload after Dispose", backend.Name, "disposed");
        ThrowsNaming<ObjectDisposedException>(() => backend.Download(kept, read), "Download after Dispose", backend.Name, "disposed");
        ThrowsNaming<ObjectDisposedException>(() => backend.DownloadRange(kept, 0, read.AsSpan(0, 4)), "DownloadRange after Dispose", backend.Name, "disposed");
        ThrowsNaming<ObjectDisposedException>(() => backend.Fill(kept, 256, 1f), "an operation (Fill) after Dispose", backend.Name, "disposed");
        backend.Return(kept);                                              // a storage outliving its backend: nothing kept
        Check(backend.GetMemoryUsage().Cached == 0, $"a Return after Dispose caches nothing: {backend.GetMemoryUsage()}");
        backend.Dispose();                                                 // twice: a no-op
        Check(backend.IsDisposed, "a second Dispose leaves it disposed, without throwing");

        // The device's own backend lives for the process: Dispose does nothing and it keeps working.
        var own = Device.Cpu.Backend;
        own.Dispose();
        Check(!own.IsDisposed, "the CPU device's own backend ignores Dispose");
        using var x = Tensor.From([1f, 2f, 3f], [3], Device.Cpu);
        Check((x + x).ToArray().SequenceEqual([2f, 4f, 6f]), "the CPU device's own backend still computes after a Dispose call");
    }

    // Sizes 200 (given back first) and 100 (given back last) cached; a miss of 150 frees the 200 block only: the cache keeps
    // the 100 block, the process holds no more than before (in use + cached), and the next 100 is a hit. Then with 1000
    // floats in use and the two sizes cached, a miss of 150 frees nothing and both hit.
    private static void CpuPoolFreesOnlyWhatAMissNeeds()
    {
        var backend = Device.Cpu.CreateSeparateBackend(new SeparateBackendOptions("pool test", 16L << 20));
        try
        {
            var old = backend.Allocate(200, zeroed: false);
            var recent = backend.Allocate(100, zeroed: false);
            backend.Return(old);
            backend.Return(recent);
            var before = backend.GetMemoryUsage();
            Check(before.Cached == 300 * sizeof(float), $"both blocks cached: {before}");
            var missed = backend.Allocate(150, zeroed: false);
            var after = backend.GetMemoryUsage();
            Check(after.Cached == 100 * sizeof(float), $"the miss freed the least recently used size only (the 100-float block stays cached): {after}");
            Check(after.InUse + after.Cached <= before.InUse + before.Cached, $"in use plus cached did not grow: before {before}, after {after}");
            var again = backend.Allocate(100, zeroed: false);
            Check(backend.GetMemoryUsage().Cached == 0, $"the size still in use is a hit: {backend.GetMemoryUsage()}");
            backend.Return(missed);
            backend.Return(again);

            // With more in use than cached (a model's weights, the live activations), a miss frees nothing: the cycle's
            // sizes, given back, all hit on the next round.
            backend.ReleaseCachedMemory();
            var held = backend.Allocate(1000, zeroed: false);
            var a = backend.Allocate(200, zeroed: false);
            var b = backend.Allocate(100, zeroed: false);
            backend.Return(a);
            backend.Return(b);
            var cycled = backend.GetMemoryUsage();
            var fresh = backend.Allocate(150, zeroed: false);
            Check(backend.GetMemoryUsage().Cached == cycled.Cached, $"a miss while the pool caches less than is in use frees nothing: before {cycled}, after {backend.GetMemoryUsage()}");
            var a2 = backend.Allocate(200, zeroed: false);
            var b2 = backend.Allocate(100, zeroed: false);
            Check(backend.GetMemoryUsage().Cached == 0, $"both cached sizes hit after the miss: {backend.GetMemoryUsage()}");
            backend.Return(held);
            backend.Return(fresh);
            backend.Return(a2);
            backend.Return(b2);
        }
        finally
        {
            backend.Dispose();
        }
    }

    private static void SeparateGpuBackendReturnsMemory(Device device)
    {
        if (device.Type == DeviceType.Cpu || device.Kind is not ("cuda" or "hip" or "vulkan"))
        {
            Skip($"a GPU's driver memory: checked on cuda, hip and vulkan (not on {device})");
        }

        const int Cycles = 6;
        const int Floats = 16 << 20;                                       // 64 MiB a cycle: a leak shows past any granularity

        // A device the driver reports as a CPU (a software Vulkan driver) has no memory of its own: its "free memory" is
        // the host's free memory, which every process on the machine moves, so it measures nothing about this backend.
        // The cycles and the refusals after each dispose still run.
        string? unmeasured = device.Kind == "vulkan" && VulkanBackend.DeviceKind(device.Ordinal).Type == VulkanDriver.DeviceTypeCpu
            ? "the Vulkan driver reports a CPU device, whose free memory is the host's (moved by every process)"
            : (device.Backend as IMemoryReport)?.FreeMemory() is null ? $"{device.Backend.Name} reports no free memory from its driver" : null;
        if (unmeasured is not null)
        {
            for (int i = 0; i <= Cycles; i++)
            {
                Cycle(device, Floats, i);
            }

            Console.WriteLine($"    {Cycles + 1} separate backends of {device} created, used and disposed, each refusing calls after; free memory not measured: {unmeasured}");
            return;
        }

        var report = (IMemoryReport)device.Backend;
        // One cycle first, so what the driver keeps for the process (its context, the first module loads) is in the start.
        Cycle(device, Floats, 0);
        long start = report.FreeMemory()!.Value;
        long lowest = start;
        for (int i = 1; i <= Cycles; i++)
        {
            Cycle(device, Floats, i);
            lowest = Math.Min(lowest, report.FreeMemory()!.Value);
        }

        long end = report.FreeMemory()!.Value;
        long lost = start - end;
        const long Granularity = 8L << 20;                                 // the driver's pages and allocation rounding
        Check(lost <= Granularity, $"{Cycles} separate backends of {device} created, used ({Floats * 4L >> 20} MiB each) and disposed lose {lost >> 20} MiB of free memory "
                                   + $"(start {start >> 20} MiB, end {end >> 20} MiB, lowest {lowest >> 20} MiB); at most {Granularity >> 20} MiB is the driver's rounding");
    }

    // One separate backend: allocate, compute, read back, dispose, then a call fails naming it.
    private static void Cycle(Device device, int floats, int index)
    {
        Backend backend;
        try
        {
            backend = device.CreateSeparateBackend(new SeparateBackendOptions($"backend dispose test {index}", null));
        }
        catch (NotSupportedException e)
        {
            Skip($"{device} makes no separate backend: {e.Message}");
            return;
        }

        var storage = backend.Allocate(floats, zeroed: false);
        backend.Fill(storage, floats, 3f);
        float[] tail = new float[4];
        backend.DownloadRange(storage, floats - 4, tail);
        Check(tail.All(v => v == 3f), $"{backend.Name} computes before it is disposed: [{string.Join(", ", tail)}]");
        backend.Synchronize();
        backend.Dispose();
        Check(backend.IsDisposed, $"{backend.Name} is disposed");
        ThrowsNaming<ObjectDisposedException>(() => backend.Allocate(16, zeroed: false), $"{backend.Name}: Allocate after Dispose", backend.Name, "disposed");
        ThrowsNaming<ObjectDisposedException>(() => backend.Fill(storage, 16, 1f), $"{backend.Name}: an operation after Dispose", backend.Name, "disposed");
        backend.Return(storage);                                           // outlived its backend: a no-op, never a throw
        backend.Dispose();                                                 // twice: a no-op
    }
}
