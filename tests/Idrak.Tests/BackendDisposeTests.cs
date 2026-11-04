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
// call after that fails naming it, the device's own backend lives for the process, and a simulated cluster gives its
// ranks' carved devices back. On a GPU, creating and disposing a separate backend again and again leaves the device's
// free memory where it started (rule 77).
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] BackendDisposeGroup =
    [
        ("backend dispose: a separate CPU backend gives its pool back; Allocate, Upload, Download and an operation after it throw ObjectDisposedException naming the backend; Return is a no-op and a second Dispose does nothing; the device's own backend ignores Dispose", d => { CpuOnly(d); SeparateCpuBackendDisposes(); }),
        ("backend dispose: a simulated cluster disposes every rank's carved backend when the run ends (counted), a failed rank's as it leaves, a tensor kept past the run fails naming its device; ReleaseDevices=false keeps them until SimulatedDevices.Release", d => { CpuOnly(d); SimulatedClusterReleasesDevices(); }),
        ("backend dispose: a carved device released before its first use refuses to start a backend, naming the release; Release refuses a device that is not carved, and carving alike afterwards gives a new device", d => { CpuOnly(d); ReleasedCarvedDevice(); }),
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

    private static void SimulatedClusterReleasesDevices()
    {
        // Three ranks carved from the CPU; each keeps a tensor past the run and records its backend.
        var topology = DpTopology(Device.Cpu, 3);
        var backends = new Backend?[3];
        var kept = new Tensor?[3];
        var devices = new Device?[3];
        InProcessCluster.Run(topology, DpOptions(), rank =>
        {
            devices[rank.Rank] = rank.Device;
            backends[rank.Rank] = rank.Device.Backend;
            kept[rank.Rank] = Tensor.From([1f, 2f, 3f, 4f], [4], rank.Device);
            Check(kept[rank.Rank]!.Sum().ToArray()[0] == 10f, $"rank {rank.Rank} computes on its carved device");
        }).ThrowIfFailed();
        int disposed = backends.Count(b => b?.IsDisposed == true);
        Check(devices.All(d => d?.Kind == SimulatedDevices.Kind), $"every rank runs on a carved device: {string.Join(", ", devices.Select(d => d?.ToString()))}");
        Check(disposed == 3, $"the run disposes each of its 3 ranks' backends as it ends: {disposed} disposed");
        ThrowsNaming<ObjectDisposedException>(() => kept[1]!.ToArray(), "reading a tensor kept past the run", backends[1]!.Name, "disposed");
        ThrowsNaming<ObjectDisposedException>(() => _ = kept[2]! + kept[2]!, "computing on a tensor kept past the run", "disposed");

        // ReleaseDevices off: the ranks' devices stay until the caller releases them.
        var keep = DpOptions() with { Simulation = new SimulationOptions { ReleaseDevices = false } };
        var held = new Backend?[2];
        var heldDevices = new Device?[2];
        var heldTensors = new Tensor?[2];
        InProcessCluster.Run(DpTopology(Device.Cpu, 2), keep, rank =>
        {
            heldDevices[rank.Rank] = rank.Device;
            held[rank.Rank] = rank.Device.Backend;
            heldTensors[rank.Rank] = Tensor.From([5f, 6f], [2], rank.Device);
        }).ThrowIfFailed();
        Check(held.All(b => b is { IsDisposed: false }), "with ReleaseDevices=false the ranks' backends outlive the run");
        Check(heldTensors[0]!.ToArray().SequenceEqual([5f, 6f]), "with ReleaseDevices=false a tensor kept past the run still reads");
        foreach (var device in heldDevices)
        {
            SimulatedDevices.Release(device!);
        }

        Check(held.All(b => b is { IsDisposed: true }), "SimulatedDevices.Release disposes each kept device's backend");
        SimulatedDevices.Release(heldDevices[0]!);                         // released twice: forgotten, not an error

        // A rank that fails gives its device back as it leaves, while the others still run.
        var leaving = new Backend?[2];
        bool sawReleased = false;
        var failed = InProcessCluster.Run(DpTopology(Device.Cpu, 2), DpOptions(), rank =>
        {
            Volatile.Write(ref leaving[rank.Rank], rank.Device.Backend);
            if (rank.Rank == 1)
            {
                throw new InvalidOperationException("rank 1 leaves on purpose");
            }

            var clock = Stopwatch.StartNew();
            while (Volatile.Read(ref leaving[1]) is not { IsDisposed: true } && clock.Elapsed < TimeSpan.FromSeconds(60))
            {
                Thread.Sleep(5);
            }

            sawReleased = Volatile.Read(ref leaving[1]) is { IsDisposed: true };
            Check(!rank.Device.Backend.IsDisposed, "rank 0's backend lives while rank 0 runs");
        });
        Check(failed.Errors[1] is InvalidOperationException { Message: "rank 1 leaves on purpose" }, $"rank 1's own error is the run's: {failed.Errors[1]}");
        Check(failed.Errors[0] is null, $"rank 0 runs on after rank 1 leaves: {failed.Errors[0]}");
        Check(sawReleased, "a failed rank's backend is disposed as it leaves, before the run ends");
        Check(leaving.All(b => b is { IsDisposed: true }), "every rank's backend is disposed once the failed run ends");
    }

    private static void ReleasedCarvedDevice()
    {
        var first = SimulatedDevices.Carve(Device.Cpu, 8L << 20, slot: 9_000);
        SimulatedDevices.Release(first);
        ThrowsNaming<ObjectDisposedException>(() => _ = first.Backend, "a released carved device's first use", first.ToString(), "released");
        ThrowsNaming<ArgumentException>(() => SimulatedDevices.Release(Device.Cpu), "releasing the CPU", "not a carved device");

        var again = SimulatedDevices.Carve(Device.Cpu, 8L << 20, slot: 9_000);
        Check(again != first, $"carving alike after a release gives a new device: {first} then {again}");
        using (var x = Tensor.From([1f, 2f], [2], again))
        {
            Check((x * 2f).ToArray().SequenceEqual([2f, 4f]), "the newly carved device computes");
        }

        Check(SimulatedDevices.Carve(Device.Cpu, 8L << 20, slot: 9_000) == again, "carving alike while it is held gives the same device back");
        SimulatedDevices.Release(again);
        Check(again.Backend.IsDisposed, "releasing a started carved device disposes its backend");
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
